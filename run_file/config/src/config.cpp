//
// Created by yu on 2025/4/3.
//
#include "config.hpp"

namespace YAML
{
    template<>
    struct convert<CONFIG::PARAMETERS>
    {
        static bool decode(const Node &node, CONFIG::PARAMETERS &rhs)
        {
            if (!node.IsMap()) return false;
            rhs.xml_path = node["model_xml_path"].as<string>();
            rhs.bin_path = node["model_bin_path"].as<string>();
            rhs.img_size = node["input_size"].as<int>();

            return true;
        };
    };


}

void CONFIG::ReadConfigParameters()
{
    YAML::Node configNode = YAML::LoadFile("../config/config.yaml");
    _config._parameters = configNode.as<CONFIG::PARAMETERS>();
}
