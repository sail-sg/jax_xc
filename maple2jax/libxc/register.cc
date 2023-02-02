// Copyright 2022 Garena Online Private Limited
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "register.h"  // NOLINT

std::map<init_fn, get_param_fn> registry;
std::map<void*, std::string> work_to_maple_name;
