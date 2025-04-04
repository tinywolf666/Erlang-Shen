#pragma once

#include <iostream>
#include <yaml-cpp/yaml.h>

using namespace std;



class CONFIG
{
public:
    struct PARAMETERS
    {
        string xml_path;
        string bin_path;
        int img_size;
    } _parameters;

    void ReadConfigParameters();
} inline _config;