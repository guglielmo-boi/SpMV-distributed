#ifndef MTX_PARSER_HPP
#define MTX_PARSER_HPP

#include "common.hpp"

#include <vector>
#include <string>
#include <map>

class MtxParser
{
public:
    struct MtxMatrix
    {
        int rows;
        int cols;
        int nnz;

        std::map<int, std::map<int, dtype>> elements; // [row, [col, value]]
    };
    
    static MtxParser::MtxMatrix parseMtxFile(const std::string& file_path);
};

#endif