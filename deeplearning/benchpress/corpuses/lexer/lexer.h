// coding=utf-8
// Copyright 2022 Foivos Tsimpourlas and Chris Cummins.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <vector>

#include "deeplearning/benchpress/proto/internal.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

#include "labm8/cpp/string.h"

namespace benchpress {

// Determine if any of a set of strings starts with prefix.
// This assumes that strings set and prefix are not empty.
bool HasPrefix(const absl::flat_hash_set<string>& strings,
               const std::string& prefix);

// Determine if any of a s et of strings matches a string.
// This assumes that strings set and match are note empty.
bool HasMatch(const absl::flat_hash_set<string>& strings,
              const std::string& match);

// Tokenize a string into a list of tokens, where candidate_vocabulary is the
// set of multi-character tokens, and vocabulary is a dictionary of mappings
// from token to indices array.
std::vector<int> TokenizeInput(
    const std::string& input,
    const absl::flat_hash_set<string>& candidate_vocabulary,
    absl::flat_hash_map<string, int>* vocabulary);

// Process a single LexerJob inplace.
void ProcessLexerJob(LexerJob* input,
                     const absl::flat_hash_set<string>& candidate_vocabulary,
                     absl::flat_hash_map<string, int>* vocabulary);

// Process a LexerBatchJob. Any errors will lead to fatal program crash.
void ProcessLexerBatchJobOrDie(LexerBatchJob* proto);

}  // namespace benchpress
