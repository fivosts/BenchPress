//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass collects the count of all instructions and reports them
//
//===----------------------------------------------------------------------===//

// #include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "InstCount"
#define FEATURES_SIZE 70

namespace {

  std::array<std::pair<std::string, int>, FEATURES_SIZE> instcount = {{
    {"TotalInsts", 0},
    {"TotalBlocks", 0},
    {"TotalFuncs", 0},
    #define HANDLE_INST(N, OPCODE, CLASS) {#OPCODE, 0},
    #include "llvm/IR/Instruction.def"
  }};

  class InstCount : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    InstCount() : ModulePass(ID) {}
    bool runOnModule(Module &M) override;
  };
}

char InstCount::ID = 0;
static RegisterPass<InstCount> X("InstCount", "Counts the various types of Instructions",
                             false /* Only looks at CFG */,
                             true /* Analysis Pass */);

bool InstCount::runOnModule(Module &M) {
  for (auto &F : M) {
    if (!F.isDeclaration()) {
      instcount[2].second++;
    }
    for (auto &BB : F){
      instcount[1].second++;
      for (auto &I : BB){
        instcount[0].second++;

        instcount[2 + I.getOpcode()].second++;
      }
    }
  }
  for (auto p : instcount){
    outs() << p.first << " : " << p.second << "\n";
  }
  return false;
}
