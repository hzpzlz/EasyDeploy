#pragma once

#ifndef __ED_PARAMS_H__
#define __ED_PARAMS_H__

#include "ed_engine_base.h"
#include "inference.h"
#include "yaml.h"

int getConfigs(InferConfig config, EdModelInfo& model_info);

#endif
