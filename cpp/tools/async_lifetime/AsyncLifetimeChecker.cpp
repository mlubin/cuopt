/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SummaryInference.hpp"

#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
#include <clang/StaticAnalyzer/Core/BugReporter/BugType.h>
#include <clang/StaticAnalyzer/Core/Checker.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h>
#include <clang/StaticAnalyzer/Core/PathSensitive/Store.h>
#include <clang/StaticAnalyzer/Frontend/CheckerRegistry.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace clang;
using namespace ento;

namespace {

struct borrow_t {
  const MemRegion* region;
  SVal stream;
  unsigned sequence;
  bool write;
  const Stmt* origin;
  const MemRegion* owner;

  bool operator==(const borrow_t& other) const = default;
  void Profile(llvm::FoldingSetNodeID& id) const
  {
    id.AddPointer(region);
    stream.Profile(id);
    id.AddInteger(sequence);
    id.AddBoolean(write);
    id.AddPointer(origin);
    id.AddPointer(owner);
  }
};

struct fence_t {
  SVal stream;
  unsigned sequence;
  bool operator==(const fence_t& other) const = default;
  void Profile(llvm::FoldingSetNodeID& id) const
  {
    stream.Profile(id);
    id.AddInteger(sequence);
  }
};

struct dependency_t {
  SVal destination;
  SVal source;
  unsigned source_sequence;
  unsigned sequence;
  bool operator==(const dependency_t& other) const = default;
  void Profile(llvm::FoldingSetNodeID& id) const
  {
    destination.Profile(id);
    source.Profile(id);
    id.AddInteger(source_sequence);
    id.AddInteger(sequence);
  }
};

std::string qualified_name(const NamedDecl* declaration)
{
  if (!declaration) return {};
  if (const auto* constructor = dyn_cast<CXXConstructorDecl>(declaration)) {
    return qualified_name(constructor->getParent());
  }
  // RMM and libcu++ use versioned inline namespaces. Those are ABI details,
  // whereas contracts name the public declaration and primary template.
  std::vector<std::string> parts;
  for (const auto* context = declaration->getDeclContext(); context;
       context             = context->getParent()) {
    if (const auto* ns = dyn_cast<NamespaceDecl>(context)) {
      if (!ns->isInline() && !ns->isAnonymousNamespace()) parts.push_back(ns->getNameAsString());
    } else if (const auto* record = dyn_cast<RecordDecl>(context)) {
      parts.push_back(record->getNameAsString());
    }
  }
  std::string result;
  for (auto it = parts.rbegin(); it != parts.rend(); ++it)
    result += *it + "::";
  result += declaration->getNameAsString();
  // CUDA's per-thread/default-stream entry points have the same storage
  // contracts as their public API names. Do not normalize other namespaces.
  if (result.starts_with("cuda") && result.find("::") == std::string::npos &&
      (result.ends_with("_ptsz") || result.ends_with("_ptds"))) {
    result.resize(result.size() - 5);
  }
  return result;
}

const Expr* unwrap(const Expr* expression)
{
  while (expression) {
    expression = expression->IgnoreParenImpCasts();
    if (const auto* clean = dyn_cast<ExprWithCleanups>(expression)) {
      expression = clean->getSubExpr();
    } else if (const auto* temporary = dyn_cast<MaterializeTemporaryExpr>(expression)) {
      expression = temporary->getSubExpr();
    } else if (const auto* binding = dyn_cast<CXXBindTemporaryExpr>(expression)) {
      expression = binding->getSubExpr();
    } else {
      break;
    }
  }
  return expression;
}

bool overlaps(const MemRegion* lhs, const MemRegion* rhs)
{
  if (!lhs || !rhs) return false;
  return lhs == rhs || lhs->isSubRegionOf(rhs) || rhs->isSubRegionOf(lhs);
}

const MemRegion* storage_root(const MemRegion* region)
{
  // Array element regions may have different symbolic indices. Conservatively
  // treat accesses to the same backing array as overlapping.
  while (const auto* element = dyn_cast_or_null<ElementRegion>(region)) {
    region = element->getSuperRegion();
  }
  return region;
}

struct argument_t {
  unsigned index;
  std::string storage;
  bool write;
};

struct contract_t {
  std::string id;
  std::string name;
  std::string kind;
  std::string effect;
  std::vector<argument_t> borrows;
  std::vector<std::pair<unsigned, std::string>> argument_types;
  int stream_arg{-1};
  bool receiver_stream{false};
  bool all_streams{false};
  bool success_return{false};
  unsigned minimum{0};
  unsigned maximum{~0u};
  int ownership_arg{-1};
  int output_arg{-1};
  int required_nonnull{-1};
  bool stream_ordered_free{false};
  std::string unsupported_reason;
  bool inferred{false};
};

}  // namespace

REGISTER_LIST_WITH_PROGRAMSTATE(AsyncBorrows, borrow_t)
REGISTER_LIST_WITH_PROGRAMSTATE(AsyncDependencies, dependency_t)
REGISTER_MAP_WITH_PROGRAMSTATE(AsyncEvents, const MemRegion*, fence_t)
REGISTER_MAP_WITH_PROGRAMSTATE(AsyncContainerStorage, const MemRegion*, const MemRegion*)
REGISTER_MAP_WITH_PROGRAMSTATE(AsyncDeviceAllocation, const MemRegion*, bool)
REGISTER_TRAIT_WITH_PROGRAMSTATE(AsyncSequence, unsigned)

namespace {

class async_lifetime_checker_t : public Checker<check::PreCall,
                                                check::PostCall,
                                                check::Bind,
                                                check::PreStmt<CXXDeleteExpr>,
                                                check::PreStmt<ReturnStmt>,
                                                check::Location,
                                                check::LifetimeEnd,
                                                check::EndFunction,
                                                check::EndAnalysis,
                                                check::EndOfTranslationUnit> {
  mutable std::unique_ptr<BugType> bug_type;
  mutable bool initialized{false};
  mutable std::vector<contract_t> contracts;
  mutable llvm::StringMap<unsigned> modeled;
  mutable llvm::StringMap<unsigned> uncovered;
  mutable unsigned reports{0};
  mutable unsigned analyses{0};
  mutable unsigned cutoffs{0};
  mutable std::set<std::pair<const Stmt*, const MemRegion*>> emitted;
  mutable std::string coverage_path;

  void incomplete(llvm::StringRef reason) const { ++uncovered[reason]; }

  void initialize(AnalysisManager& manager) const
  {
    if (initialized) return;
    initialized   = true;
    auto& options = manager.getAnalyzerOptions();
    coverage_path = options.getCheckerStringOption(this, "Coverage").str();
    auto path     = options.getCheckerStringOption(this, "Contracts");
    auto buffer   = llvm::MemoryBuffer::getFile(path);
    if (!buffer) llvm::report_fatal_error("Cannot read asynchronous lifetime contracts: " + path);
    auto parsed = llvm::json::parse(buffer.get()->getBuffer());
    if (!parsed) llvm::report_fatal_error("Invalid asynchronous lifetime contracts JSON");
    const auto* root = parsed->getAsObject();
    if (!root || root->getInteger("schema_version") != 1) {
      llvm::report_fatal_error("Unsupported asynchronous lifetime contract schema");
    }
    llvm::json::Array entries;
    for (auto section : {"contracts", "unsupported"}) {
      const auto* values = root->getArray(section);
      if (!values)
        llvm::report_fatal_error("Contract registry requires contracts and unsupported arrays");
      for (const auto& entry : *values)
        entries.push_back(entry);
    }
    auto inferred = cuopt_async_lifetime::infer_wrapper_contracts(
      manager.getASTContext().getTranslationUnitDecl(), entries);
    for (auto& entry : inferred.contracts) {
      auto* model = entry.getAsObject();
      model->insert({"id", "inferred: " + model->getString("name").value_or("").str()});
      entries.push_back(std::move(entry));
    }
    // An incomplete inference is scoped to calls of that wrapper. Merely having
    // a complex function definition in the translation unit is not a coverage gap.
    for (auto& entry : inferred.incomplete)
      entries.push_back(std::move(entry));
    auto invalid = [](const std::string& reason) -> void {
      llvm::report_fatal_error(llvm::Twine("Invalid asynchronous lifetime contract: ") + reason);
    };
    auto keys = [&](const llvm::json::Object& object,
                    std::initializer_list<llvm::StringRef> allowed,
                    llvm::StringRef location) {
      for (const auto& field : object) {
        const std::string key = field.first.str();
        if (std::find(allowed.begin(), allowed.end(), llvm::StringRef(key)) == allowed.end()) {
          invalid("unrecognized " + location.str() + " field: " + key);
        }
      }
    };
    auto index = [&](const llvm::json::Object& object, llvm::StringRef key, int fallback = -1) {
      const auto* value = object.get(key);
      if (!value) return fallback;
      const auto integer = value->getAsInteger();
      if (!integer || *integer < 0 || *integer > std::numeric_limits<int>::max()) {
        invalid("argument index must be a nonnegative integer: " + key.str());
      }
      return static_cast<int>(*integer);
    };
    auto boolean = [&](
                     const llvm::json::Object& object, llvm::StringRef key, bool fallback = false) {
      const auto* value = object.get(key);
      if (!value) return fallback;
      const auto flag = value->getAsBoolean();
      if (!flag) invalid("expected boolean: " + key.str());
      return *flag;
    };
    auto string =
      [&](const llvm::json::Object& object, llvm::StringRef key, llvm::StringRef fallback = "") {
        const auto* value = object.get(key);
        if (!value) return fallback.str();
        const auto text = value->getAsString();
        if (!text) invalid("expected string: " + key.str());
        return text->str();
      };
    for (const auto& entry : entries) {
      const auto* object = entry.getAsObject();
      if (!object) invalid("contract must be an object");
      keys(*object,
           {"name",
            "id",
            "kind",
            "effect",
            "min_args",
            "max_args",
            "success_return",
            "reason",
            "ownership",
            "requires",
            "stream",
            "when",
            "borrows",
            "description",
            "evidence",
            "project_usage",
            "runtime_contract",
            "return_borrow",
            "retained_by",
            "lifetime",
            "mode_dependent_borrows",
            "state",
            "inferred",
            "signature",
            "file",
            "line"},
           "contract");
      contract_t contract;
      contract.name   = string(*object, "name");
      contract.id     = string(*object, "id", contract.name);
      contract.kind   = string(*object, "kind", "function");
      contract.effect = string(*object, "effect");
      if (contract.name.empty() || contract.effect.empty()) {
        invalid("name and effect are required");
      }
      if (contract.kind != "function" && contract.kind != "method" &&
          contract.kind != "constructor") {
        invalid("unrecognized kind: " + contract.kind);
      }
      const std::set<std::string> effects{"async",
                                          "synchronize",
                                          "retain",
                                          "state",
                                          "allocate",
                                          "deallocate",
                                          "immediate",
                                          "unsupported"};
      if (!effects.contains(contract.effect)) invalid("unrecognized effect: " + contract.effect);
      contract.minimum = index(*object, "min_args", 0);
      if (object->get("max_args")) contract.maximum = index(*object, "max_args");
      if (contract.maximum < contract.minimum) invalid("max_args precedes min_args");
      contract.success_return     = boolean(*object, "success_return");
      contract.inferred           = boolean(*object, "inferred");
      contract.unsupported_reason = string(*object, "reason");
      auto unsupported            = [&](llvm::StringRef reason) {
        if (!contract.unsupported_reason.empty()) contract.unsupported_reason += "; ";
        contract.unsupported_reason += reason.str();
      };
      if (contract.effect == "unsupported" && contract.unsupported_reason.empty()) {
        invalid("unsupported effect requires a reason");
      }
      if (object->get("return_borrow")) unsupported("returned allocation borrow is not modeled");
      if (object->get("mode_dependent_borrows")) unsupported("handle pointer mode is not modeled");
      if (object->get("runtime_contract") &&
          string(*object, "runtime_contract") != "cuda_memcpy_direction_and_memory_kind")
        unsupported("unrecognized runtime completion contract");
      // The CUDA memcpy model deliberately conservatively keeps both buffers
      // borrowed. It never assumes pageable staging implies completion.
      if (const auto* ownership = object->getObject("ownership")) {
        keys(*ownership,
             {"arg", "out_arg", "receiver", "result", "storage", "source", "deallocation"},
             "ownership");
        if (ownership->get("receiver")) boolean(*ownership, "receiver");
        if (ownership->get("result")) string(*ownership, "result");
        if (ownership->get("source")) string(*ownership, "source");
        contract.ownership_arg = index(*ownership, "arg");
        contract.output_arg    = index(*ownership, "out_arg");
        const auto order       = string(*ownership, "deallocation");
        if (!order.empty() && order != "stream_ordered" && order != "immediate") {
          invalid("unrecognized deallocation order: " + order);
        }
        contract.stream_ordered_free = order == "stream_ordered";
        const bool plain_storage     = !ownership->get("receiver") && !ownership->get("result") &&
                                   !ownership->get("source") &&
                                   string(*ownership, "storage", "pointee") == "pointee";
        const bool raw_deallocation = contract.effect == "deallocate" &&
                                      contract.ownership_arg >= 0 && !ownership->get("out_arg");
        const bool raw_allocation =
          contract.effect == "allocate" && contract.output_arg >= 0 && !ownership->get("arg");
        if (!plain_storage || (!raw_deallocation && !raw_allocation)) {
          unsupported("allocation/result/container ownership is not modeled");
        }
      } else if (object->get("ownership")) {
        invalid("ownership must be an object");
      }
      if (const auto* prerequisites = object->getObject("requires")) {
        keys(*prerequisites, {"nonnull_arg", "nonnull_container_data_arg"}, "requires");
        contract.required_nonnull = index(*prerequisites, "nonnull_arg");
        if (prerequisites->get("nonnull_container_data_arg")) {
          index(*prerequisites, "nonnull_container_data_arg");
          unsupported("container nullness precondition unresolved");
        }
        if (contract.effect != "synchronize")
          unsupported("non-synchronization precondition is not modeled");
      } else if (object->get("requires")) {
        invalid("requires must be an object");
      }
      if (const auto* stream = object->getObject("stream")) {
        keys(*stream,
             {"arg", "receiver", "all", "handle_arg", "receiver_member", "event_arg"},
             "stream");
        if (stream->size() != 1) invalid("stream requires exactly one selector");
        contract.stream_arg      = index(*stream, "arg");
        contract.receiver_stream = boolean(*stream, "receiver");
        contract.all_streams     = boolean(*stream, "all");
        if ((stream->get("receiver") && !contract.receiver_stream) ||
            (stream->get("all") && !contract.all_streams))
          invalid("stream selector must be true");
        if (stream->get("handle_arg") || stream->get("receiver_member")) {
          if (stream->get("handle_arg")) index(*stream, "handle_arg");
          if (stream->get("receiver_member")) string(*stream, "receiver_member");
          unsupported("unresolved handle/member stream contract");
        }
        if (stream->get("event_arg") &&
            (index(*stream, "event_arg") != 0 || contract.effect != "synchronize" ||
             (contract.name != "cudaEventSynchronize" && contract.name != "cudaEventQuery"))) {
          unsupported("unrecognized event synchronization contract");
        }
      } else if (object->get("stream")) {
        invalid("stream must be an object");
      } else if (contract.effect == "async" || contract.effect == "synchronize") {
        unsupported("asynchronous operation has no modeled stream");
      }
      if (const auto* when = object->getObject("when")) {
        keys(*when, {"arg_types"}, "when");
        if (const auto* types = when->getObject("arg_types")) {
          for (const auto& type : *types) {
            unsigned position;
            if (llvm::StringRef(type.first.str()).getAsInteger(10, position) ||
                !type.second.getAsString()) {
              invalid("argument type selector requires an index and string");
            }
            const auto expected = *type.second.getAsString();
            if (expected != "pointer" && expected != "non_pointer" &&
                expected != "arithmetic_or_enum" &&
                !(expected.starts_with("record:") && expected.size() > 7)) {
              unsupported("unrecognized argument type selector: " + expected.str());
            }
            contract.argument_types.emplace_back(position, expected.str());
          }
        } else
          invalid("when.arg_types must be an object");
      } else if (object->get("when")) {
        invalid("when must be an object");
      }
      if (const auto* borrows = object->getArray("borrows")) {
        for (const auto& borrow : *borrows) {
          const auto* value = borrow.getAsObject();
          if (!value) invalid("borrow must be an object");
          keys(*value, {"arg", "storage", "access", "device_only"}, "borrow");
          const auto position = index(*value, "arg");
          if (position < 0) invalid("borrow requires an argument index");
          const auto storage = string(*value, "storage", "pointee");
          const auto access  = string(*value, "access", "read");
          if (storage != "referent" && storage != "pointee" && storage != "container_data") {
            invalid("unrecognized borrow storage: " + storage);
          }
          if (access != "read" && access != "write" && access != "read_write") {
            invalid("unrecognized borrow access: " + access);
          }
          // Raw pointers use conservative pending-access tracking regardless
          // of memory kind. Owning device containers also need allocation
          // identity and stream-ordered destruction, which are not inferred.
          if (boolean(*value, "device_only") && storage != "pointee") {
            unsupported("device container allocation identity/lifetime is not modeled");
          }
          contract.borrows.push_back({static_cast<unsigned>(position), storage, access != "read"});
        }
      } else if (object->get("borrows")) {
        invalid("borrows must be an array");
      }
      if (const auto* owner = object->getObject("retained_by")) {
        keys(*owner, {"receiver", "arg", "out_arg"}, "retained_by");
        if (owner->size() != 1 || !boolean(*owner, "receiver")) {
          unsupported("non-receiver retention owner is not modeled");
        }
        if (owner->get("arg")) index(*owner, "arg");
        if (owner->get("out_arg")) index(*owner, "out_arg");
      } else if (object->get("retained_by")) {
        invalid("retained_by must be an object");
      } else if (contract.effect == "retain") {
        unsupported("retained storage has no modeled owner");
      }
      if (contract.effect == "retain" && contract.kind != "method") {
        unsupported("retention receiver requires an instance method");
      }
      if (const auto* state = object->getObject("state")) {
        keys(*state,
             {"operation",
              "family",
              "handle_arg",
              "mode_arg",
              "event_arg",
              "input_mode",
              "result_mode"},
             "state");
        const auto operation = string(*state, "operation");
        const bool record =
          operation == "record_event" && index(*state, "event_arg") == 0 &&
          (contract.name == "cudaEventRecord" || contract.name == "cudaEventRecordWithFlags") &&
          contract.stream_arg == 1;
        const bool wait = operation == "wait_event" && index(*state, "event_arg") == 1 &&
                          contract.name == "cudaStreamWaitEvent" && contract.stream_arg == 0;
        if (contract.effect != "state" || (!record && !wait))
          unsupported("handle or library state transition is not modeled");
      } else if (object->get("state")) {
        invalid("state must be an object");
      } else if (contract.effect == "state") {
        unsupported("state effect has no modeled transition");
      }
      if (contract.effect == "allocate" && contract.output_arg < 0) {
        unsupported("allocation output identity is not modeled");
      }
      if (contract.effect == "deallocate" && contract.ownership_arg < 0) {
        unsupported("deallocation has no modeled allocation argument");
      }
      if (contract.effect == "immediate") {
        unsupported("immediate API side effects have not been audited by the engine");
      }
      contracts.push_back(std::move(contract));
    }
  }

  bool matches(const contract_t& contract, const CallEvent& call) const
  {
    const auto* declaration = dyn_cast_or_null<NamedDecl>(call.getDecl());
    if (qualified_name(declaration) != contract.name) return false;
    const auto* method = dyn_cast_or_null<CXXMethodDecl>(declaration);
    if ((contract.kind == "constructor") != isa_and_nonnull<CXXConstructorDecl>(declaration))
      return false;
    if (contract.kind == "method" &&
        (!method || isa<CXXConstructorDecl, CXXDestructorDecl>(method)))
      return false;
    if (contract.kind == "function" && method) return false;
    if (call.getNumArgs() < contract.minimum || call.getNumArgs() > contract.maximum) return false;
    if (contract.stream_arg >= 0 && static_cast<unsigned>(contract.stream_arg) >= call.getNumArgs())
      return false;
    if (contract.required_nonnull >= 0 &&
        static_cast<unsigned>(contract.required_nonnull) >= call.getNumArgs())
      return false;
    for (const auto& borrow : contract.borrows)
      if (borrow.index >= call.getNumArgs()) return false;
    for (const auto& [index, expected] : contract.argument_types) {
      if (index >= call.getNumArgs() || !call.getArgExpr(index)) return false;
      QualType type = call.getArgExpr(index)->getType().getNonReferenceType();
      if (expected == "pointer" && !type->isPointerType()) return false;
      if (expected == "non_pointer" && type->isPointerType()) return false;
      if (expected == "arithmetic_or_enum" && !type->isArithmeticType() &&
          !type->isEnumeralType()) {
        return false;
      }
      if (llvm::StringRef(expected).starts_with("record:")) {
        const auto* record = type->getAsCXXRecordDecl();
        if (!record || qualified_name(record) != llvm::StringRef(expected).drop_front(7))
          return false;
      }
    }
    return true;
  }

  const contract_t* lookup(const CallEvent& call) const
  {
    for (const auto& contract : contracts) {
      if (matches(contract, call)) return &contract;
    }
    return nullptr;
  }

  SVal stream_record(SVal value, QualType type, ProgramStateRef state) const
  {
    type = type.getNonReferenceType();
    if (type->isPointerType()) type = type->getPointeeType();
    const auto* record = type->getAsCXXRecordDecl();
    auto name          = qualified_name(record);
    if (!record || (name != "cuda::stream_ref" && name != "rmm::cuda_stream_view")) {
      return UnknownVal();
    }
    record = record->getDefinition();
    if (!record || record->getNumBases() != 0) return UnknownVal();
    // These audited non-owning wrappers contain one native stream pointer.
    // Other layouts (including owning streams) need their own models.
    const FieldDecl* native = nullptr;
    for (const auto* field : record->fields()) {
      if (native || !field->getType()->isPointerType()) return UnknownVal();
      native = field;
    }
    if (!native) return UnknownVal();
    auto& regions = state->getStateManager().getRegionManager();
    if (auto lazy = value.getAs<nonloc::LazyCompoundVal>()) {
      // A LazyCompoundVal is a frozen value, not the current storage of its
      // backing region. Read its field from that snapshot's store.
      const auto* field = regions.getFieldRegion(native, lazy->getRegion());
      return state->getStateManager().getStoreManager().getBinding(
        lazy->getStore(), loc::MemRegionVal(field), native->getType());
    }
    const auto* base = dyn_cast_or_null<SubRegion>(value.getAsRegion());
    if (!base) return UnknownVal();
    return state->getSVal(regions.getFieldRegion(native, base), native->getType());
  }

  SVal stream_value(const Expr* expression,
                    ProgramStateRef state,
                    const StackFrame* context,
                    unsigned depth = 0) const
  {
    if (!expression || depth > 12) return UnknownVal();
    const Expr* evaluated = expression;
    expression            = unwrap(expression);
    if (const auto* construction = dyn_cast<CXXConstructExpr>(expression)) {
      auto name = qualified_name(construction->getConstructor()->getParent());
      if (construction->getNumArgs() == 1 &&
          (name == "cuda::stream_ref" || name == "rmm::cuda_stream_view")) {
        // The evaluated argument is a frozen copy. Its source local can already
        // be dead (or later reassigned), so prefer the actual passed value.
        auto value = state->getSVal(evaluated, context);
        if (!value.isUnknownOrUndef()) {
          return stream_record(value, construction->getType(), state);
        }
        return stream_value(construction->getArg(0), state, context, depth + 1);
      }
    }
    if (const auto* method = dyn_cast<CXXMemberCallExpr>(expression)) {
      auto name = qualified_name(method->getMethodDecl());
      if (name == "cuda::stream_ref::get" || name == "rmm::cuda_stream_view::value") {
        auto value = state->getSVal(evaluated, context);
        if (!value.isUnknownOrUndef()) return value;
        return stream_value(method->getImplicitObjectArgument(), state, context, depth + 1);
      }
    }
    // Consult the current symbolic store, rather than a local's initializer:
    // assignment and opaque mutation can change a stream wrapper's identity.
    auto value = state->getSVal(evaluated, context);
    auto type  = expression->getType().getNonReferenceType();
    if (value.isUnknownOrUndef()) {
      // The analyzer may discard the evaluated lvalue after a trivial copy
      // constructor. Resolve the variable's current storage, never its initializer.
      if (const auto* reference = dyn_cast<DeclRefExpr>(expression)) {
        if (const auto* variable = dyn_cast<VarDecl>(reference->getDecl())) {
          value = state->getLValue(variable, context);
          if (variable->getType()->isReferenceType() || !type->isRecordType()) {
            if (auto location = value.getAs<Loc>()) value = state->getSVal(*location);
          }
        }
      }
    }
    if (type->isRecordType() ||
        (type->isPointerType() && type->getPointeeType()->isRecordType() &&
         (qualified_name(type->getPointeeType()->getAsCXXRecordDecl()) == "cuda::stream_ref" ||
          qualified_name(type->getPointeeType()->getAsCXXRecordDecl()) ==
            "rmm::cuda_stream_view"))) {
      return stream_record(value, type, state);
    }
    return value;
  }

  SVal stream_for(const contract_t& contract, const CallEvent& call, CheckerContext& context) const
  {
    if (contract.stream_arg >= 0 &&
        static_cast<unsigned>(contract.stream_arg) < call.getNumArgs()) {
      return stream_value(
        call.getArgExpr(contract.stream_arg), context.getState(), context.getStackFrame());
    }
    if (contract.receiver_stream) {
      if (const auto* member = dyn_cast_or_null<CXXMemberCallExpr>(call.getOriginExpr())) {
        return stream_value(
          member->getImplicitObjectArgument(), context.getState(), context.getStackFrame());
      }
    }
    return UnknownVal();
  }

  const MemRegion* region_for(const argument_t& argument,
                              const CallEvent& call,
                              CheckerContext& context) const
  {
    if (argument.index >= call.getNumArgs()) return nullptr;
    SVal value = call.getArgSVal(argument.index);
    if (argument.storage == "container_data") {
      const auto* owner = value.getAsRegion();
      if (const auto lazy = value.getAs<nonloc::LazyCompoundVal>()) owner = lazy->getRegion();
      if (owner) {
        if (const auto* allocation = context.getState()->get<AsyncContainerStorage>(owner)) {
          return *allocation;
        }
      }
      return nullptr;
    }
    if (const auto lazy = value.getAs<nonloc::LazyCompoundVal>()) return lazy->getRegion();
    if (const auto* region = value.getAsRegion()) return storage_root(region);
    if (argument.storage == "referent") {
      auto* expression = call.getArgExpr(argument.index);
      if (expression) {
        auto location = context.getState()->getSVal(expression, context.getStackFrame());
        if (const auto* region = location.getAsRegion()) return storage_root(region);
        if (const auto* reference = dyn_cast<DeclRefExpr>(unwrap(expression))) {
          if (const auto* variable = dyn_cast<VarDecl>(reference->getDecl())) {
            return context.getSValBuilder().getRegionManager().getVarRegion(
              variable, context.getStackFrame());
          }
        }
      }
    }
    return nullptr;
  }

  const MemRegion* receiver(const CallEvent& call) const
  {
    if (const auto* instance = dyn_cast<CXXInstanceCall>(&call)) {
      return instance->getCXXThisVal().getAsRegion();
    }
    if (const auto* constructor = dyn_cast<AnyCXXConstructorCall>(&call)) {
      return constructor->getCXXThisVal().getAsRegion();
    }
    return nullptr;
  }

  bool vector_call(const CallEvent& call) const
  {
    const auto* method = dyn_cast_or_null<CXXMethodDecl>(call.getDecl());
    return method && qualified_name(method->getParent()) == "std::vector";
  }

  ProgramStateRef release_container(ProgramStateRef state,
                                    const MemRegion* owner,
                                    CheckerContext& context) const
  {
    const auto* allocation = state->get<AsyncContainerStorage>(owner);
    if (!allocation) return state;
    for (const auto& borrow : state->get<AsyncBorrows>()) {
      if (overlaps(*allocation, borrow.region)) {
        report("Container releases storage while an asynchronous access is pending",
               borrow,
               state,
               context);
      }
    }
    return without_region(state->remove<AsyncContainerStorage>(owner), *allocation);
  }

  bool container_pre_call(const CallEvent& call, CheckerContext& context) const
  {
    if (!vector_call(call)) return false;
    const auto* method = cast<CXXMethodDecl>(call.getDecl());
    const auto* owner  = receiver(call);
    if (!owner) {
      incomplete("unresolved std::vector owner");
      return true;
    }
    auto state = context.getState();
    if (isa<CXXDestructorDecl>(method)) {
      auto next = release_container(state, owner, context);
      if (next != state) context.addTransition(next);
      return true;
    }
    if (call.getNumArgs() && method->getNumParams() &&
        method->getParamDecl(0)->getType()->isRValueReferenceType() &&
        (isa<CXXConstructorDecl>(method) || method->isMoveAssignmentOperator())) {
      const auto* source = call.getArgSVal(0).getAsRegion();
      if (const auto lazy = call.getArgSVal(0).getAs<nonloc::LazyCompoundVal>()) {
        source = lazy->getRegion();
      }
      const auto* specialization = dyn_cast<ClassTemplateSpecializationDecl>(method->getParent());
      bool standard_allocator =
        specialization && specialization->getTemplateArgs().size() == 2 &&
        qualified_name(specialization->getTemplateArgs()[1].getAsType()->getAsCXXRecordDecl()) ==
          "std::allocator";
      if (!source || !standard_allocator) {
        incomplete("std::vector move requires resolved storage and std::allocator");
        return true;
      }
      state = release_container(state, owner, context);
      if (const auto* allocation = state->get<AsyncContainerStorage>(source)) {
        state = state->set<AsyncContainerStorage>(owner, *allocation);
        state = state->remove<AsyncContainerStorage>(source);
      }
      context.addTransition(state);
      return true;
    }
    auto name = method->getNameAsString();
    if (name != "data" && name != "size" && name != "clear" && !isa<CXXConstructorDecl>(method) &&
        state->get<AsyncContainerStorage>(owner)) {
      // Capacity-changing operations may invalidate pointers. They need their
      // own conditional reallocation models before they can be certified.
      incomplete("unmodeled std::vector storage operation: " + name);
    }
    return true;
  }

  void report(llvm::StringRef message,
              const borrow_t& borrow,
              ProgramStateRef state,
              CheckerContext& context) const
  {
    if (!emitted.emplace(borrow.origin, borrow.region).second) return;
    if (!bug_type)
      bug_type = std::make_unique<BugType>(this, "asynchronous storage lifetime", "cuOpt");
    auto* node = context.generateNonFatalErrorNode(state);
    if (!node) return;
    auto diagnostic = std::make_unique<PathSensitiveBugReport>(*bug_type, message, node);
    if (borrow.origin) {
      auto position = PathDiagnosticLocation::createBegin(
        borrow.origin, context.getSourceManager(), context.getStackFrame());
      diagnostic->addNote("Storage borrowed here by an asynchronous operation", position);
    }
    diagnostic->markInteresting(borrow.region);
    context.emitReport(std::move(diagnostic));
    ++reports;
  }

  ProgramStateRef without_region(ProgramStateRef state, const MemRegion* region) const
  {
    auto previous = state->get<AsyncBorrows>();
    state         = state->remove<AsyncBorrows>();
    for (const auto& borrow : previous) {
      if (!overlaps(region, borrow.region) && !overlaps(region, borrow.owner)) {
        state = state->add<AsyncBorrows>(borrow);
      }
    }
    return state;
  }

  ProgramStateRef complete(ProgramStateRef state,
                           SVal stream,
                           unsigned sequence,
                           unsigned depth = 0) const
  {
    if (stream.isUnknownOrUndef()) {
      incomplete("synchronization stream could not be resolved");
      return state;
    }
    if (depth > 16) {
      incomplete("stream dependency depth limit");
      return state;
    }
    auto previous = state->get<AsyncBorrows>();
    state         = state->remove<AsyncBorrows>();
    for (const auto& borrow : previous) {
      if (borrow.owner || borrow.stream != stream || borrow.sequence > sequence) {
        state = state->add<AsyncBorrows>(borrow);
      }
    }
    for (const auto& dependency : state->get<AsyncDependencies>()) {
      if (dependency.destination == stream && dependency.sequence <= sequence) {
        state = complete(state, dependency.source, dependency.source_sequence, depth + 1);
      }
    }
    return state;
  }

  void end_region(const MemRegion* region, CheckerContext& context) const
  {
    if (terminating_path(context)) return;
    auto state = context.getState();
    for (const auto& borrow : state->get<AsyncBorrows>()) {
      if (!overlaps(region, borrow.region)) continue;
      // If the retaining object dies in the same scope first, no future use is
      // possible through that object. Its own queued accesses remain separate.
      report("Storage lifetime ends while an asynchronous or retained access is pending",
             borrow,
             state,
             context);
    }
    auto next = without_region(state, region);
    if (next != state) context.addTransition(next);
  }

  bool terminating_path(CheckerContext& context) const
  {
    const auto* block = context.getCFGElementRef().getParent();
    if (!block || !block->hasNoReturnElement()) return false;
    for (const auto& element : *block) {
      auto statement = element.getAs<CFGStmt>();
      if (!statement) continue;
      const auto* call     = dyn_cast<CallExpr>(statement->getStmt());
      const auto* function = call ? call->getDirectCallee() : nullptr;
      if (!function) continue;
      auto name = function->getNameAsString();
      if (name == "__builtin_unreachable" || name == "abort" || name == "_Exit" ||
          name == "_exit" || name == "terminate")
        return true;
    }
    return false;
  }

 public:
  void checkPreCall(const CallEvent& call, CheckerContext& context) const
  {
    initialize(context.getAnalysisManager());
    if (container_pre_call(call, context)) return;
    if (isa_and_nonnull<CUDAKernelCallExpr>(call.getOriginExpr())) {
      incomplete("CUDA kernel launch buffer lifetimes require a launch contract");
      return;
    }
    if (const auto* function = dyn_cast_or_null<FunctionDecl>(call.getDecl())) {
      if (function->isReplaceableGlobalAllocationFunction() ||
          function->isReservedGlobalPlacementOperator()) {
        if (function->getOverloadedOperator() == OO_New ||
            function->getOverloadedOperator() == OO_Array_New)
          return;
      }
      if (function->isOverloadedOperator() && call.getNumArgs() > 0 &&
          (function->getOverloadedOperator() == OO_Delete ||
           function->getOverloadedOperator() == OO_Array_Delete)) {
        release_host(call.getArgSVal(0), context);
        return;
      }
    }
    const auto* contract = lookup(call);
    auto name            = qualified_name(dyn_cast_or_null<NamedDecl>(call.getDecl()));
    if (!contract) {
      const auto* declaration = dyn_cast_or_null<FunctionDecl>(call.getDecl());
      if (!declaration || (!declaration->hasBody() && !declaration->getBuiltinID())) {
        if (!context.getState()->get<AsyncBorrows>().isEmpty()) {
          // Even a zero-argument callee can mutate escaped storage through a
          // global alias. Argument overlap alone does not describe its effects.
          incomplete("opaque call with pending accesses requires an effects contract: " + name);
        }
        bool storage_interface = receiver(call) != nullptr ||
                                 call.getResultType()->isPointerType() ||
                                 call.getResultType()->isReferenceType();
        for (unsigned index = 0; index < call.getNumArgs(); ++index) {
          storage_interface |= call.getArgSVal(index).getAsRegion() != nullptr ||
                               call.getArgSVal(index).getAs<nonloc::LazyCompoundVal>().has_value();
        }
        if (storage_interface) incomplete("opaque storage interface requires a contract: " + name);
      }
      if (!call.getDecl() && !context.getState()->get<AsyncBorrows>().isEmpty()) {
        incomplete("indirect call while asynchronous accesses are pending");
      } else if (name.starts_with("cuda") || name.starts_with("rmm::") ||
                 name.starts_with("raft::") || name.starts_with("cublas") ||
                 name.starts_with("cusparse") || name.starts_with("cusolver") ||
                 name.starts_with("cub::") || name.starts_with("thrust::")) {
        // Bodies are followed by CSA. An opaque relevant call must not be
        // mistaken for a modeled no-op or a completion barrier.
        const auto* function = dyn_cast_or_null<FunctionDecl>(call.getDecl());
        if (!function || !function->hasBody()) incomplete("unmodeled API: " + name);
      }
      if (!context.getState()->get<AsyncBorrows>().isEmpty()) {
        const auto* function = dyn_cast_or_null<FunctionDecl>(call.getDecl());
        if (!function || !function->hasBody()) {
          for (unsigned index = 0; index < call.getNumArgs(); ++index) {
            auto* argument = storage_root(call.getArgSVal(index).getAsRegion());
            for (const auto& borrow : context.getState()->get<AsyncBorrows>()) {
              if (overlaps(argument, borrow.region)) {
                incomplete("opaque call may access pending asynchronous storage: " + name);
              }
            }
          }
        }
      }
      return;
    }
    if (contract->effect == "unsupported" || !contract->unsupported_reason.empty()) {
      incomplete(contract->name + ": " + contract->unsupported_reason);
      return;
    }
    // Use an inferred summary only when CSA skips its implementation.
    if (contract->inferred) return;
    ++modeled[contract->id];
    if (contract->effect == "deallocate") {
      if (contract->ownership_arg < 0 ||
          static_cast<unsigned>(contract->ownership_arg) >= call.getNumArgs()) {
        incomplete("deallocation argument is not modeled: " + name);
        return;
      }
      auto* region = storage_root(call.getArgSVal(contract->ownership_arg).getAsRegion());
      if (!region) {
        incomplete("unresolved allocation deallocation: " + name);
        return;
      }
      auto state  = context.getState();
      auto stream = stream_for(*contract, call, context);
      for (const auto& borrow : state->get<AsyncBorrows>()) {
        if (!overlaps(region, borrow.region)) continue;
        if (contract->stream_ordered_free && !borrow.owner && !stream.isUnknownOrUndef() &&
            borrow.stream == stream)
          continue;
        report("Storage released before an asynchronous access is ordered to complete",
               borrow,
               state,
               context);
      }
      return;
    }
    if (contract->effect != "async" && contract->effect != "retain") return;
    apply_borrows(*contract, call, context);
  }

  void apply_borrows(const contract_t& model, const CallEvent& call, CheckerContext& context) const
  {
    const auto* contract = &model;
    auto name            = qualified_name(dyn_cast_or_null<NamedDecl>(call.getDecl()));
    auto state           = context.getState();
    auto stream          = stream_for(*contract, call, context);
    auto owner           = contract->effect == "retain" ? receiver(call) : nullptr;
    if (contract->effect == "retain" && !owner) incomplete("unresolved retention owner: " + name);
    if (contract->effect == "async" && stream.isUnknownOrUndef()) {
      incomplete("unresolved asynchronous stream: " + name);
    }
    unsigned sequence = state->get<AsyncSequence>() + 1;
    state             = state->set<AsyncSequence>(sequence);
    for (const auto& argument : contract->borrows) {
      if (argument.index < call.getNumArgs() && argument.storage != "referent" &&
          call.getArgSVal(argument.index).isZeroConstant())
        continue;
      const auto* region = region_for(argument, call, context);
      if (!region) {
        incomplete("unresolved borrowed storage: " + name);
        continue;
      }
      borrow_t borrow{region, stream, sequence, argument.write, call.getOriginExpr(), owner};
      state = state->add<AsyncBorrows>(borrow);
    }
    context.addTransition(state);
  }

  void release_host(SVal pointer, CheckerContext& context) const
  {
    const auto* region = storage_root(pointer.getAsRegion());
    if (!region) return;
    auto state = context.getState();
    for (const auto& borrow : state->get<AsyncBorrows>()) {
      if (overlaps(region, borrow.region)) {
        report("Storage released while an asynchronous or retained access is pending",
               borrow,
               state,
               context);
      }
    }
    auto next = without_region(state, region);
    if (next != state) context.addTransition(next);
  }

  void checkPreStmt(const CXXDeleteExpr* expression, CheckerContext& context) const
  {
    release_host(context.getSVal(expression->getArgument()), context);
  }

  void checkPreStmt(const ReturnStmt* statement, CheckerContext& context) const
  {
    // NRVO and implicit moves transfer a vector's allocation to the caller.
    // The allocation remains borrowed; only the local ownership association ends.
    const auto* variable = statement->getNRVOCandidate();
    if (!variable || qualified_name(variable->getType()->getAsCXXRecordDecl()) != "std::vector")
      return;
    const auto* owner =
      context.getSValBuilder().getRegionManager().getVarRegion(variable, context.getStackFrame());
    auto state = context.getState();
    if (state->get<AsyncContainerStorage>(owner)) {
      context.addTransition(state->remove<AsyncContainerStorage>(owner));
    }
  }

  ProgramStateRef successful_ownership(const CallEvent& call,
                                       const contract_t& contract,
                                       CheckerContext& context) const
  {
    auto state = context.getState();
    if (!contract.success_return) return state;
    auto result = call.getReturnValue().getAs<DefinedOrUnknownSVal>();
    if (!result || result->isUnknown()) {
      incomplete("unresolved allocation/deallocation status");
      return nullptr;
    }
    auto predicate = context.getSValBuilder().evalEQ(
      state, *result, context.getSValBuilder().makeZeroVal(call.getResultType()));
    auto condition = predicate.getAs<DefinedOrUnknownSVal>();
    if (!condition) {
      incomplete("unresolved allocation/deallocation status");
      return nullptr;
    }
    auto branches = state->assume(*condition);
    if (branches.second) context.addTransition(branches.second);
    return branches.first;
  }

  ProgramStateRef successful_event_call(const CallEvent& call, CheckerContext& context) const
  {
    auto state  = context.getState();
    auto result = call.getReturnValue().getAs<DefinedOrUnknownSVal>();
    if (!result || result->isUnknown()) {
      incomplete("unresolved CUDA event status");
      return nullptr;
    }
    auto predicate = context.getSValBuilder().evalEQ(
      state, *result, context.getSValBuilder().makeZeroVal(call.getResultType()));
    auto condition = predicate.getAs<DefinedOrUnknownSVal>();
    if (!condition) {
      incomplete("unresolved CUDA event status");
      return nullptr;
    }
    auto branches = state->assume(*condition);
    // An unsuccessful record must preserve the previous event generation; an
    // unsuccessful wait must not install a dependency that never existed.
    if (branches.second) context.addTransition(branches.second);
    return branches.first;
  }

  void checkPostCall(const CallEvent& call, CheckerContext& context) const
  {
    const auto* contract = lookup(call);
    auto name            = qualified_name(dyn_cast_or_null<NamedDecl>(call.getDecl()));
    auto state           = context.getState();
    if (contract && contract->inferred && contract->unsupported_reason.empty() &&
        contract->effect == "async") {
      if (!context.wasInlined) {
        ++modeled[contract->id];
        apply_borrows(*contract, call, context);
      }
      return;
    }
    if (contract && contract->unsupported_reason.empty() &&
        (contract->effect == "allocate" || contract->effect == "deallocate")) {
      auto success = successful_ownership(call, *contract, context);
      if (!success) return;
      const MemRegion* allocation = nullptr;
      if (contract->effect == "allocate" && contract->output_arg >= 0) {
        if (auto output = call.getArgSVal(contract->output_arg).getAs<Loc>()) {
          allocation = storage_root(success->getSVal(*output).getAsRegion());
        }
        if (allocation) success = success->set<AsyncDeviceAllocation>(allocation, true);
      } else if (contract->effect == "deallocate" && contract->ownership_arg >= 0) {
        allocation = storage_root(call.getArgSVal(contract->ownership_arg).getAsRegion());
        if (allocation) {
          success = without_region(success, allocation)->remove<AsyncDeviceAllocation>(allocation);
        }
      }
      if (!allocation) incomplete("unresolved allocation identity: " + name);
      context.addTransition(success);
      return;
    }
    if (vector_call(call)) {
      const auto* method = cast<CXXMethodDecl>(call.getDecl());
      if (method->getNameAsString() == "data") {
        const auto* owner      = receiver(call);
        const auto* allocation = storage_root(call.getReturnValue().getAsRegion());
        if (owner && allocation) {
          context.addTransition(state->set<AsyncContainerStorage>(owner, allocation));
        } else {
          incomplete("unresolved std::vector backing allocation");
        }
      }
      return;
    }
    if (!contract && !context.wasInlined) {
      const auto* function = dyn_cast_or_null<FunctionDecl>(call.getDecl());
      const bool library_storage =
        name.starts_with("rmm::") || name.starts_with("raft::") || name.starts_with("cublas") ||
        name.starts_with("cusparse") || name.starts_with("cusolver") || name.starts_with("cub::") ||
        name.starts_with("thrust::") || name.starts_with("std::unique_ptr::") ||
        name.starts_with("std::shared_ptr::") || name.starts_with("std::basic_string::") ||
        name.starts_with("std::async") || name.starts_with("std::thread::") ||
        name.starts_with("std::jthread::");
      if (function && function->hasBody() && !function->getLocation().isInvalid() &&
          !function->isDefaulted() &&
          (library_storage ||
           !context.getSourceManager().isInSystemHeader(function->getLocation()))) {
        incomplete("call body was not analyzed and has no contract: " + name);
      }
    }
    // Event records and waits represent GPU ordering, not host completion.
    if ((name == "cudaEventRecord" || name == "cudaEventRecordWithFlags") &&
        call.getNumArgs() >= 2) {
      if (!contract || contract->effect == "unsupported" || !contract->unsupported_reason.empty())
        return;
      state = successful_event_call(call, context);
      if (!state) return;
      auto event = call.getArgSVal(0).getAsRegion();
      if (!event) {
        incomplete("unresolved CUDA event record");
        context.addTransition(state);
        return;
      }
      if (name == "cudaEventRecordWithFlags" && call.getNumArgs() >= 3 &&
          !call.getArgSVal(2).isZeroConstant()) {
        incomplete("CUDA event record flags require a capture-aware model");
        context.addTransition(state);
        return;
      }
      auto stream = stream_value(call.getArgExpr(1), state, context.getStackFrame());
      if (stream.isUnknownOrUndef()) {
        incomplete("unresolved CUDA event recording stream");
        context.addTransition(state);
        return;
      }
      context.addTransition(state->set<AsyncEvents>(event, {stream, state->get<AsyncSequence>()}));
      return;
    }
    if (name == "cudaStreamWaitEvent" && call.getNumArgs() >= 2) {
      if (!contract || contract->effect == "unsupported" || !contract->unsupported_reason.empty())
        return;
      state = successful_event_call(call, context);
      if (!state) return;
      const auto* event = call.getArgSVal(1).getAsRegion();
      const auto* fence = event ? state->get<AsyncEvents>(event) : nullptr;
      if (!fence) {
        incomplete("unresolved CUDA event dependency");
        context.addTransition(state);
        return;
      }
      if (call.getNumArgs() >= 3 && !call.getArgSVal(2).isZeroConstant()) {
        incomplete("CUDA event wait flags require a capture-aware model");
        context.addTransition(state);
        return;
      }
      auto stream = stream_value(call.getArgExpr(0), state, context.getStackFrame());
      if (stream.isUnknownOrUndef() || fence->stream.isUnknownOrUndef()) {
        incomplete("unresolved CUDA event dependency stream");
        context.addTransition(state);
        return;
      }
      auto sequence = state->get<AsyncSequence>() + 1;
      state         = state->set<AsyncSequence>(sequence);
      context.addTransition(
        state->add<AsyncDependencies>({stream, fence->stream, fence->sequence, sequence}));
      return;
    }
    if ((name == "cudaEventSynchronize" || name == "cudaEventQuery") && call.getNumArgs() >= 1) {
      if (!contract || contract->effect == "unsupported" || !contract->unsupported_reason.empty())
        return;
      const auto* event = call.getArgSVal(0).getAsRegion();
      const auto* fence = event ? state->get<AsyncEvents>(event) : nullptr;
      if (!fence) {
        incomplete("unresolved CUDA event synchronization");
        return;
      }
      synchronize(call, context, fence->stream, fence->sequence, true, false);
      return;
    }
    if (!contract || contract->effect != "synchronize" || !contract->unsupported_reason.empty())
      return;
    if (contract->required_nonnull >= 0) {
      auto value = call.getArgSVal(contract->required_nonnull).getAs<DefinedOrUnknownSVal>();
      if (!value) {
        incomplete("unresolved synchronization precondition: " + name);
        return;
      }
      auto branches = state->assume(*value);
      if (branches.second) {
        context.addTransition(branches.second);
        if (!branches.first) return;
        // A possible null early return must not discharge other operations.
        incomplete("conditional synchronization precondition: " + name);
        return;
      }
    }
    synchronize(call,
                context,
                stream_for(*contract, call, context),
                state->get<AsyncSequence>(),
                contract->success_return,
                contract->all_streams);
  }

  void synchronize(const CallEvent& call,
                   CheckerContext& context,
                   SVal stream,
                   unsigned sequence,
                   bool check_result,
                   bool all) const
  {
    auto state              = context.getState();
    ProgramStateRef success = state;
    if (check_result) {
      auto result = call.getReturnValue().getAs<DefinedOrUnknownSVal>();
      if (!result || result->isUnknown()) {
        incomplete("unresolved synchronization status");
        return;
      }
      auto predicate = context.getSValBuilder().evalEQ(
        state, *result, context.getSValBuilder().makeZeroVal(call.getResultType()));
      auto condition = predicate.getAs<DefinedOrUnknownSVal>();
      if (!condition) {
        incomplete("unresolved synchronization status");
        return;
      }
      auto branches = state->assume(*condition);
      success       = branches.first;
      if (branches.second) context.addTransition(branches.second);
    }
    if (!success) return;
    if (all) {
      incomplete("device-wide completion requires current-device and stream-device identity");
    } else {
      success = complete(success, stream, sequence);
    }
    context.addTransition(success);
  }

  void checkBind(
    SVal location, SVal, const Stmt*, bool at_declaration, CheckerContext& context) const
  {
    if (at_declaration) return;
    auto* region = storage_root(location.getAsRegion());
    if (!region) return;
    for (const auto& borrow : context.getState()->get<AsyncBorrows>()) {
      if (!borrow.owner && overlaps(region, borrow.region)) {
        report("Storage modified while an asynchronous access is pending",
               borrow,
               context.getState(),
               context);
      }
    }
  }

  void checkLocation(SVal location, bool load, const Stmt*, CheckerContext& context) const
  {
    if (!load) return;
    auto* region = storage_root(location.getAsRegion());
    if (!region) return;
    for (const auto& borrow : context.getState()->get<AsyncBorrows>()) {
      if (borrow.write && overlaps(region, borrow.region)) {
        report("Storage read before an asynchronous write is known to have completed",
               borrow,
               context.getState(),
               context);
      }
    }
  }

  void checkLifetimeEnd(const VarDecl* variable, CheckerContext& context) const
  {
    if (!variable->hasLocalStorage() || variable->getType()->isReferenceType()) return;
    auto* region =
      context.getSValBuilder().getRegionManager().getVarRegion(variable, context.getStackFrame());
    if (auto next = release_container(context.getState(), region, context);
        next != context.getState())
      context.addTransition(next);
    end_region(region, context);
  }

  void checkEndFunction(const ReturnStmt*, CheckerContext& context) const
  {
    auto state        = context.getState();
    const auto* frame = context.getStackFrame();
    std::vector<const MemRegion*> ending;
    for (const auto& borrow : state->get<AsyncBorrows>()) {
      const auto* memory = borrow.region->getMemorySpace(state);
      const auto* stack  = dyn_cast<StackSpaceRegion>(memory);
      if (borrow.owner) {
        const auto* owner_stack = dyn_cast<StackSpaceRegion>(borrow.owner->getMemorySpace(state));
        if (owner_stack && owner_stack->getStackFrame() == frame) continue;
      }
      if (stack && stack->getStackFrame() == frame) ending.push_back(borrow.region);
    }
    for (const auto* region : ending)
      end_region(region, context);
  }

  void checkEndAnalysis(ExplodedGraph&, BugReporter&, ExprEngine& engine) const
  {
    ++analyses;
    if (engine.hasWorkRemaining()) {
      ++cutoffs;
      incomplete("analyzer exploration limit reached");
    }
  }

  void checkEndOfTranslationUnit(const TranslationUnitDecl*,
                                 AnalysisManager& manager,
                                 BugReporter&) const
  {
    initialize(manager);
    llvm::json::Object known;
    llvm::json::Object unknown;
    for (const auto& entry : modeled)
      known[entry.first()] = entry.second;
    for (const auto& entry : uncovered)
      unknown[entry.first()] = entry.second;
    llvm::json::Object output{{"schema_version", 1},
                              {"analyses", analyses},
                              {"diagnostics", reports},
                              {"cutoffs", cutoffs},
                              {"modeled_calls", std::move(known)},
                              {"incomplete", std::move(unknown)}};
    if (coverage_path.empty()) {
      if (!uncovered.empty()) {
        llvm::errs() << "cuopt.AsyncLifetime: analysis incomplete (" << uncovered.size()
                     << " reasons); use the Coverage option for details\n";
      }
      return;
    }
    std::error_code error;
    llvm::raw_fd_ostream file(coverage_path, error);
    if (error) llvm::report_fatal_error("Cannot write asynchronous lifetime coverage report");
    file << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(output)));
  }
};

}  // namespace

extern "C" void clang_registerCheckers(CheckerRegistry& registry)
{
  registry.addChecker<async_lifetime_checker_t>(
    "cuopt.AsyncLifetime", "Check storage borrowed by asynchronous and retained operations", "");
  registry.addCheckerOption("string",
                            "cuopt.AsyncLifetime",
                            "Contracts",
                            "",
                            "Path to the versioned API contract registry",
                            "released");
  registry.addCheckerOption("string",
                            "cuopt.AsyncLifetime",
                            "Coverage",
                            "",
                            "Write machine-readable analysis coverage to this file",
                            "released");
  registry.addDependency("cuopt.AsyncLifetime", "core.builtin.BuiltinFunctions");
  registry.addDependency("cuopt.AsyncLifetime", "core.builtin.NoReturnFunctions");
}

extern "C" const char clang_analyzerAPIVersionString[] = CLANG_ANALYZER_API_VERSION_STRING;
