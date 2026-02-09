#pragma once

#include "ive/AST.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/Support/CommandLine.h>

#include <string>

namespace cl = llvm::cl;

enum InputType { Ive, MLIR };

enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};

namespace ive {

/// Returns a Ive AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<ive::ModuleAST> parseInputFile(llvm::StringRef filename);
int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module,
             cl::opt<std::string> &inputFileName,
             cl::opt<enum InputType> &inputType);
int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module,
                       cl::opt<std::string> &inputFileName,
                       cl::opt<enum InputType> &inputType,
                       cl::opt<enum Action> &emitAction,
                       cl::opt<bool> &enableOpt);
int dumpAST(cl::opt<std::string> &inputFileName,
            cl::opt<enum InputType> &inputType);
int dumpLLVMIR(mlir::ModuleOp module, cl::opt<bool> &enableOpt);
int runJit(mlir::ModuleOp module, cl::opt<bool> &enableOpt);

} // namespace ive
