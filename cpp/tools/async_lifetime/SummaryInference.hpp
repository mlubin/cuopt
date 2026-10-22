/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/SourceManager.h>
#include <llvm/Support/JSON.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cuopt_async_lifetime {

struct InferredContracts {
  llvm::json::Array contracts;
  llvm::json::Array incomplete;
};

namespace summary_detail {

// Match semantic declarations, not spellings, macros, or a function's short name.
inline std::string name_of(const clang::NamedDecl* declaration)
{
  if (!declaration) return {};
  if (const auto* constructor = llvm::dyn_cast<clang::CXXConstructorDecl>(declaration)) {
    return name_of(constructor->getParent());
  }
  std::vector<std::string> components;
  for (const auto* context = declaration->getDeclContext(); context;
       context             = context->getParent()) {
    if (const auto* ns = llvm::dyn_cast<clang::NamespaceDecl>(context)) {
      if (!ns->isInline() && !ns->isAnonymousNamespace())
        components.push_back(ns->getNameAsString());
    } else if (const auto* record = llvm::dyn_cast<clang::RecordDecl>(context)) {
      components.push_back(record->getNameAsString());
    }
  }
  std::string result;
  for (auto it = components.rbegin(); it != components.rend(); ++it)
    result += *it + "::";
  return result + declaration->getNameAsString();
}

inline const clang::Expr* strip(const clang::Expr* expression)
{
  while (expression) {
    expression = expression->IgnoreParenImpCasts();
    if (const auto* clean = llvm::dyn_cast<clang::ExprWithCleanups>(expression)) {
      expression = clean->getSubExpr();
    } else if (const auto* materialized =
                 llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expression)) {
      expression = materialized->getSubExpr();
    } else if (const auto* bound = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expression)) {
      expression = bound->getSubExpr();
    } else {
      break;
    }
  }
  return expression;
}

inline bool transparent_record(llvm::StringRef name)
{
  return name == "std::vector" || name == "std::span" || name == "std::basic_string" ||
         name == "std::unique_ptr" || name == "std::shared_ptr" || name == "raft::device_span" ||
         name == "raft::host_span" || name == "rmm::device_uvector" ||
         name == "rmm::device_scalar" || name == "rmm::device_buffer" ||
         name == "cuda::stream_ref" || name == "rmm::cuda_stream_view";
}

inline bool nonowning_record(llvm::StringRef name)
{
  return name == "std::span" || name == "raft::device_span" || name == "raft::host_span";
}

inline bool stream_record(llvm::StringRef name)
{
  return name == "cuda::stream_ref" || name == "rmm::cuda_stream_view";
}

struct call_site {
  const clang::FunctionDecl* function{};
  const clang::Expr* expression{};
  std::vector<const clang::Expr*> args;
  const clang::Expr* receiver{};
};

class body_visitor : public clang::RecursiveASTVisitor<body_visitor> {
  const clang::Stmt* final_return_{};

 public:
  std::vector<call_site> calls;
  std::map<const clang::VarDecl*, const clang::Expr*> aliases;
  std::string restriction;

  explicit body_visitor(const clang::Stmt* body)
  {
    if (const auto* block = llvm::dyn_cast_or_null<clang::CompoundStmt>(body)) {
      if (!block->body_empty() && llvm::isa<clang::ReturnStmt>(block->body_back())) {
        final_return_ = block->body_back();
      }
    }
  }

  bool VisitStmt(clang::Stmt* statement)
  {
    if (llvm::isa<clang::IfStmt,
                  clang::SwitchStmt,
                  clang::ForStmt,
                  clang::WhileStmt,
                  clang::DoStmt,
                  clang::CXXForRangeStmt,
                  clang::ConditionalOperator,
                  clang::BinaryConditionalOperator,
                  clang::GotoStmt,
                  clang::IndirectGotoStmt,
                  clang::CXXTryStmt,
                  clang::CXXCatchStmt,
                  clang::LambdaExpr>(statement)) {
      restriction = "control flow or deferred closure requires path-sensitive analysis";
    }
    if (llvm::isa<clang::ReturnStmt>(statement) && statement != final_return_) {
      restriction = "early or nested return prevents an unconditional wrapper summary";
    }
    if (llvm::isa<clang::AsmStmt, clang::AtomicExpr>(statement)) {
      restriction = "inline assembly or atomic effects require path-sensitive analysis";
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(statement)) {
      if (binary->isAssignmentOp())
        restriction = "mutation inside wrapper requires path-sensitive analysis";
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(statement)) {
      if (unary->isIncrementDecrementOp())
        restriction = "mutation inside wrapper requires path-sensitive analysis";
    }
    if (llvm::isa<clang::CXXNewExpr, clang::CXXDeleteExpr, clang::CXXThrowExpr>(statement)) {
      restriction =
        "allocation, deletion, or exceptional control flow requires ownership/path analysis";
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl* variable)
  {
    if (!llvm::isa<clang::ParmVarDecl>(variable) && variable->hasInit() &&
        variable->hasLocalStorage()) {
      aliases.emplace(variable->getCanonicalDecl(), variable->getInit());
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr* call)
  {
    call_site site;
    site.function   = call->getDirectCallee();
    site.expression = call;
    site.args.assign(call->arg_begin(), call->arg_end());
    if (const auto* member = llvm::dyn_cast<clang::CXXMemberCallExpr>(call)) {
      site.receiver = member->getImplicitObjectArgument();
    }
    calls.push_back(std::move(site));
    return true;
  }

  bool VisitCXXConstructExpr(clang::CXXConstructExpr* call)
  {
    call_site site;
    site.function   = call->getConstructor();
    site.expression = call;
    site.args.assign(call->arg_begin(), call->arg_end());
    calls.push_back(std::move(site));
    return true;
  }
};

inline bool matches(const llvm::json::Object& model, const call_site& call)
{
  if (!call.function || model.getString("name").value_or("") != name_of(call.function))
    return false;
  if ((model.getString("kind").value_or("function") == "constructor") !=
      llvm::isa<clang::CXXConstructorDecl>(call.function))
    return false;
  const auto count = static_cast<int64_t>(call.args.size());
  if (count < model.getInteger("min_args").value_or(0) ||
      count > model.getInteger("max_args").value_or(std::numeric_limits<int64_t>::max()))
    return false;
  if (const auto* when = model.getObject("when")) {
    const auto* types = when->getObject("arg_types");
    if (!types || when->size() != 1) return false;
    for (const auto& type : *types) {
      unsigned index;
      const auto expected = type.second.getAsString();
      if (llvm::StringRef(type.first.str()).getAsInteger(10, index) || !expected ||
          index >= call.args.size())
        return false;
      const auto actual = call.args[index]->getType().getNonReferenceType();
      if (*expected == "pointer") {
        if (!actual->isPointerType()) return false;
      } else if (*expected == "non_pointer") {
        if (actual->isPointerType()) return false;
      } else if (*expected == "arithmetic_or_enum") {
        if (!actual->isArithmeticType() && !actual->isEnumeralType()) return false;
      } else if (expected->starts_with("record:")) {
        if (name_of(actual->getAsCXXRecordDecl()) != expected->drop_front(7)) return false;
      } else {
        return false;
      }
    }
  }
  return true;
}

inline bool transparent_call(const call_site& call)
{
  if (!call.function) return false;
  const auto name = name_of(call.function);
  if (name == "std::move" || name == "std::forward" || name == "std::addressof") return true;
  if (const auto* constructor = llvm::dyn_cast<clang::CXXConstructorDecl>(call.function)) {
    return stream_record(name_of(constructor->getParent()));
  }
  const auto* method = llvm::dyn_cast<clang::CXXMethodDecl>(call.function);
  if (!method || !transparent_record(name_of(method->getParent()))) return false;
  const auto short_name = method->getNameAsString();
  return short_name == "data" || short_name == "get" || short_name == "size" ||
         short_name == "empty" || short_name == "is_empty" || short_name == "begin" ||
         short_name == "end";
}

struct mapped_argument {
  unsigned index{};
  std::string storage;
};

class parameter_mapper {
  const clang::FunctionDecl* function_;
  const std::map<const clang::VarDecl*, const clang::Expr*>& aliases_;

  // A depth bound also prevents recursive local aliases from looping forever.
  bool map_impl(const clang::Expr* raw,
                llvm::StringRef storage,
                mapped_argument& result,
                unsigned depth) const
  {
    if (!raw || depth > aliases_.size() + 12) return false;
    const auto* expression = strip(raw);
    if (const auto* reference = llvm::dyn_cast<clang::DeclRefExpr>(expression)) {
      if (const auto* parameter = llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl())) {
        for (unsigned i = 0; i < function_->getNumParams(); ++i) {
          if (function_->getParamDecl(i) != parameter) continue;
          const auto type = parameter->getType();
          // Passing a value parameter by reference borrows the callee's copy.
          // Propagating that borrow to the caller would hide the original bug.
          if (storage == "referent" && !type->isReferenceType()) return false;
          if (storage == "container_data" && !type->isReferenceType() &&
              !nonowning_record(name_of(type->getAsCXXRecordDecl())))
            return false;
          if (storage == "pointee" && !type.getNonReferenceType()->isPointerType() &&
              !type.getNonReferenceType()->isArrayType())
            return false;
          result = {i, storage.str()};
          return true;
        }
        return false;
      }
      if (const auto* variable = llvm::dyn_cast<clang::VarDecl>(reference->getDecl())) {
        const auto found = aliases_.find(variable->getCanonicalDecl());
        if (found == aliases_.end()) return false;
        // A non-reference scalar alias creates fresh storage; a pointer alias
        // still points to the original allocation.
        if (storage == "referent" && !variable->getType()->isReferenceType()) return false;
        if (storage == "container_data" && !variable->getType()->isReferenceType() &&
            !nonowning_record(name_of(variable->getType()->getAsCXXRecordDecl())))
          return false;
        return map_impl(found->second, storage, result, depth + 1);
      }
    }
    if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(expression)) {
      if (unary->getOpcode() == clang::UO_AddrOf && storage == "pointee") {
        return map_impl(unary->getSubExpr(), "referent", result, depth + 1);
      }
      if (unary->getOpcode() == clang::UO_Deref && storage == "referent") {
        return map_impl(unary->getSubExpr(), "pointee", result, depth + 1);
      }
    }
    if (const auto* index = llvm::dyn_cast<clang::ArraySubscriptExpr>(expression)) {
      if (storage == "referent") return map_impl(index->getBase(), "pointee", result, depth + 1);
    }
    if (const auto* member = llvm::dyn_cast<clang::MemberExpr>(expression)) {
      if (storage == "referent") {
        return map_impl(
          member->getBase(), member->isArrow() ? "pointee" : "referent", result, depth + 1);
      }
    }
    if (const auto* binary = llvm::dyn_cast<clang::BinaryOperator>(expression)) {
      if (storage == "pointee" &&
          (binary->getOpcode() == clang::BO_Add || binary->getOpcode() == clang::BO_Sub)) {
        if (binary->getLHS()->getType()->isPointerType())
          return map_impl(binary->getLHS(), storage, result, depth + 1);
        if (binary->getOpcode() == clang::BO_Add && binary->getRHS()->getType()->isPointerType()) {
          return map_impl(binary->getRHS(), storage, result, depth + 1);
        }
      }
    }
    if (const auto* cast = llvm::dyn_cast<clang::ExplicitCastExpr>(expression)) {
      if (storage == "pointee" && cast->getType()->isPointerType()) {
        return map_impl(cast->getSubExpr(), storage, result, depth + 1);
      }
    }
    if (const auto* call = llvm::dyn_cast<clang::CXXMemberCallExpr>(expression)) {
      const auto* method = call->getMethodDecl();
      if (method && transparent_record(name_of(method->getParent())) &&
          (method->getNameAsString() == "data" || method->getNameAsString() == "get") &&
          storage == "pointee") {
        return map_impl(call->getImplicitObjectArgument(), "container_data", result, depth + 1);
      }
    }
    if (const auto* call = llvm::dyn_cast<clang::CallExpr>(expression)) {
      const auto name = name_of(call->getDirectCallee());
      if ((name == "std::move" || name == "std::forward") && call->getNumArgs() == 1) {
        return map_impl(call->getArg(0), storage, result, depth + 1);
      }
    }
    return false;
  }

 public:
  parameter_mapper(const clang::FunctionDecl* function,
                   const std::map<const clang::VarDecl*, const clang::Expr*>& aliases)
    : function_(function), aliases_(aliases)
  {
  }

  bool map(const clang::Expr* expression, llvm::StringRef storage, mapped_argument& result) const
  {
    return map_impl(expression, storage, result, 0);
  }

  bool stream(const clang::Expr* raw, unsigned& result, unsigned depth = 0) const
  {
    if (!raw || depth > aliases_.size() + 12) return false;
    const auto* expression = strip(raw);
    if (const auto* reference = llvm::dyn_cast<clang::DeclRefExpr>(expression)) {
      if (const auto* parameter = llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl())) {
        for (unsigned i = 0; i < function_->getNumParams(); ++i) {
          if (function_->getParamDecl(i) == parameter) {
            result = i;
            return true;
          }
        }
      }
      if (const auto* variable = llvm::dyn_cast<clang::VarDecl>(reference->getDecl())) {
        const auto found = aliases_.find(variable->getCanonicalDecl());
        if (found != aliases_.end()) return stream(found->second, result, depth + 1);
      }
    }
    if (const auto* member = llvm::dyn_cast<clang::CXXMemberCallExpr>(expression)) {
      const auto* method = member->getMethodDecl();
      if (method && method->getNameAsString() == "get" &&
          stream_record(name_of(method->getParent()))) {
        return stream(member->getImplicitObjectArgument(), result, depth + 1);
      }
    }
    if (const auto* construct = llvm::dyn_cast<clang::CXXConstructExpr>(expression)) {
      if (construct->getNumArgs() == 1 &&
          stream_record(name_of(construct->getConstructor()->getParent()))) {
        return stream(construct->getArg(0), result, depth + 1);
      }
    }
    if (const auto* cast = llvm::dyn_cast<clang::ExplicitCastExpr>(expression)) {
      if (stream_record(name_of(cast->getType()->getAsCXXRecordDecl()))) {
        return stream(cast->getSubExpr(), result, depth + 1);
      }
    }
    return false;
  }
};

class function_visitor : public clang::RecursiveASTVisitor<function_visitor> {
  std::set<const clang::FunctionDecl*> seen_;

 public:
  std::vector<const clang::FunctionDecl*> functions;
  std::map<std::pair<std::string, unsigned>, std::set<const clang::FunctionDecl*>> overloads;

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitFunctionDecl(clang::FunctionDecl* function)
  {
    if (function->isImplicit() ||
        function->getASTContext().getSourceManager().isInSystemHeader(function->getLocation()))
      return true;
    overloads[{name_of(function), function->getNumParams()}].insert(function->getCanonicalDecl());
    if (!function->doesThisDeclarationHaveABody()) return true;
    if (seen_.insert(function->getCanonicalDecl()).second) functions.push_back(function);
    return true;
  }
};

inline llvm::json::Object incomplete_model(const clang::FunctionDecl* function,
                                           llvm::StringRef reason)
{
  llvm::json::Object result{
    {"name", name_of(function)},
    {"kind", llvm::isa<clang::CXXMethodDecl>(function) ? "method" : "function"},
    {"effect", "unsupported"},
    {"reason", reason.str()},
    {"min_args", static_cast<int64_t>(function->getNumParams())},
    {"max_args", static_cast<int64_t>(function->getNumParams())},
    {"inferred", true},
    {"signature", function->getType().getAsString()}};
  const auto location =
    function->getASTContext().getSourceManager().getPresumedLoc(function->getLocation());
  if (location.isValid()) {
    result["file"] = location.getFilename();
    result["line"] = static_cast<int64_t>(location.getLine());
  }
  return result;
}

}  // namespace summary_detail

// Infer only wrappers whose behavior is unconditionally representable by the
// registry. Complex wrappers stay in CSA's path analysis and receive an explicit
// coverage record. The fixed point permits wrappers of wrappers without assuming
// that unavailable, indirect, or recursive callees are safe.
inline InferredContracts infer_wrapper_contracts(const clang::TranslationUnitDecl* translation_unit,
                                                 const llvm::json::Array& seed)
{
  using namespace summary_detail;
  InferredContracts result;
  function_visitor collector;
  collector.TraverseDecl(const_cast<clang::TranslationUnitDecl*>(translation_unit));
  std::vector<const llvm::json::Object*> known;
  for (const auto& entry : seed)
    if (const auto* model = entry.getAsObject()) known.push_back(model);
  std::deque<llvm::json::Object> generated;
  std::set<const clang::FunctionDecl*> completed;
  std::map<const clang::FunctionDecl*, std::string> failures;
  std::map<const clang::FunctionDecl*, std::unique_ptr<body_visitor>> body_cache;

  for (std::size_t round = 0; round <= collector.functions.size(); ++round) {
    bool changed = false;
    for (const auto* function : collector.functions) {
      if (completed.contains(function) ||
          llvm::isa<clang::CXXConstructorDecl, clang::CXXDestructorDecl>(function))
        continue;
      // Hand-audited library and project summaries take precedence over inference.
      if (std::any_of(known.begin(), known.end(), [&](const auto* model) {
            return !model->getBoolean("inferred").value_or(false) &&
                   model->getString("name").value_or("") == name_of(function);
          }))
        continue;
      auto& cached_body = body_cache[function];
      if (!cached_body) {
        cached_body = std::make_unique<body_visitor>(function->getBody());
        cached_body->TraverseStmt(function->getBody());
      }
      const auto& body = *cached_body;
      bool interesting = false;
      std::vector<std::pair<const call_site*, const llvm::json::Object*>> operations;
      std::string failure = body.restriction;
      for (const auto& call : body.calls) {
        const llvm::json::Object* model = nullptr;
        for (const auto* candidate : known)
          if (matches(*candidate, call)) {
            model = candidate;
            break;
          }
        if (model) {
          interesting = true;
          operations.emplace_back(&call, model);
        } else if (!transparent_call(call)) {
          failure = "opaque or indirect call prevents an unconditional wrapper summary: " +
                    name_of(call.function);
        }
      }
      if (!interesting) continue;
      if (collector.overloads.at({name_of(function), function->getNumParams()}).size() > 1) {
        failure =
          "overloads or template instantiations need a more specific semantic selector before "
          "inference";
      }
      parameter_mapper mapper(function, body.aliases);
      unsigned wrapper_stream = ~0u;
      llvm::json::Array borrows;
      std::set<std::tuple<unsigned, std::string, std::string, bool>> seen_borrows;
      for (const auto& [call, model] : operations) {
        if (model->getString("effect").value_or("") != "async" || model->getObject("requires") ||
            model->getObject("return_borrow") || model->getObject("ownership") ||
            model->getObject("retained_by") || model->getObject("state") ||
            model->getArray("mode_dependent_borrows")) {
          failure =
            "synchronization, retained storage, conditional effect, or ownership needs "
            "path-sensitive analysis: " +
            name_of(call->function);
          continue;
        }
        const auto* stream    = model->getObject("stream");
        const auto stream_arg = stream ? stream->getInteger("arg") : std::nullopt;
        unsigned mapped_stream;
        if (!stream_arg || *stream_arg < 0 ||
            static_cast<uint64_t>(*stream_arg) >= call->args.size() ||
            !mapper.stream(call->args[*stream_arg], mapped_stream)) {
          failure = "asynchronous callee stream cannot be mapped to a wrapper parameter: " +
                    name_of(call->function);
          continue;
        }
        if (wrapper_stream != ~0u && wrapper_stream != mapped_stream) {
          failure =
            "wrapper submits work on multiple streams; a single-stream summary is insufficient";
          continue;
        }
        wrapper_stream = mapped_stream;
        if (const auto* sources = model->getArray("borrows")) {
          for (const auto& value : *sources) {
            const auto* source = value.getAsObject();
            const auto index   = source ? source->getInteger("arg") : std::nullopt;
            mapped_argument mapped;
            const auto storage =
              source ? source->getString("storage").value_or("pointee") : llvm::StringRef{};
            if (!index || *index < 0 || static_cast<uint64_t>(*index) >= call->args.size() ||
                !mapper.map(call->args[*index], storage, mapped)) {
              failure =
                "asynchronous storage is local, a by-value copy, or cannot be mapped to a wrapper "
                "parameter: " +
                name_of(call->function);
              continue;
            }
            const auto access = source->getString("access").value_or("read").str();
            const bool device = source->getBoolean("device_only").value_or(false);
            if (seen_borrows.emplace(mapped.index, mapped.storage, access, device).second) {
              llvm::json::Object borrow{{"arg", static_cast<int64_t>(mapped.index)},
                                        {"storage", mapped.storage},
                                        {"access", access}};
              if (device) borrow["device_only"] = true;
              borrows.push_back(std::move(borrow));
            }
          }
        }
      }
      if (!failure.empty() || wrapper_stream == ~0u) {
        failures[function] = failure.empty() ? "no representable asynchronous stream" : failure;
        continue;
      }
      llvm::json::Object types;
      for (unsigned i = 0; i < function->getNumParams(); ++i) {
        const auto type = function->getParamDecl(i)->getType().getNonReferenceType();
        std::string selector;
        if (type->isPointerType())
          selector = "pointer";
        else if (type->isArithmeticType() || type->isEnumeralType())
          selector = "arithmetic_or_enum";
        else if (const auto* record = type->getAsCXXRecordDecl())
          selector = "record:" + name_of(record);
        else {
          failure = "dependent or unsupported parameter type prevents an exact overload selector";
          break;
        }
        types[std::to_string(i)] = selector;
      }
      if (!failure.empty()) {
        failures[function] = failure;
        continue;
      }
      llvm::json::Object inferred{
        {"name", name_of(function)},
        {"kind", llvm::isa<clang::CXXMethodDecl>(function) ? "method" : "function"},
        {"effect", "async"},
        {"inferred", true},
        {"min_args", static_cast<int64_t>(function->getNumParams())},
        {"max_args", static_cast<int64_t>(function->getNumParams())},
        {"when", llvm::json::Object{{"arg_types", std::move(types)}}},
        {"stream", llvm::json::Object{{"arg", static_cast<int64_t>(wrapper_stream)}}},
        {"borrows", std::move(borrows)},
        {"description",
         "Inferred from unconditional direct parameter forwarding; complex effects remain coverage "
         "gaps."}};
      const auto location =
        function->getASTContext().getSourceManager().getPresumedLoc(function->getLocation());
      if (location.isValid()) {
        inferred["file"] = location.getFilename();
        inferred["line"] = static_cast<int64_t>(location.getLine());
      }
      generated.push_back(std::move(inferred));
      known.push_back(&generated.back());
      completed.insert(function);
      failures.erase(function);
      changed = true;
    }
    if (!changed) break;
  }
  for (auto& model : generated)
    result.contracts.push_back(std::move(model));
  for (const auto& [function, reason] : failures)
    result.incomplete.push_back(incomplete_model(function, reason));
  return result;
}

}  // namespace cuopt_async_lifetime
