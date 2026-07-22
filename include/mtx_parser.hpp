#ifndef MTX_PARSER_HPP
#define MTX_PARSER_HPP

#include "common.hpp"

#include <vector>
#include <string>
#include <map>

class MtxParser
{
public:
    struct COOEntry 
    {
        COOEntry(int row, int col, dtype value);

        int row;
        int col;
        dtype value;

        friend bool operator<(const MtxParser::COOEntry& lhs, const MtxParser::COOEntry& rhs);
    };

    struct MtxMatrix
    {
        int rows = 0;
        int cols = 0;
        int nnz = 0;

        std::vector<COOEntry> entries;
    };
    
    static MtxParser::MtxMatrix parseMtxFile(const std::string& file_path);
};

bool operator<(const MtxParser::COOEntry& lhs, const MtxParser::COOEntry& rhs);

#endif