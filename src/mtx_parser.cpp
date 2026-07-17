#include "mtx_parser.hpp"

#include <fstream>
#include <sstream>

MtxParser::MtxMatrix MtxParser::parseMtxFile(const std::string& file_path) {
    MtxParser::MtxMatrix ret;
    std::string header, object, format, field, symmetry;

    std::ifstream ifs(file_path);

    std::string line;
    bool parsed_first_line = false;

    while (std::getline(ifs, line)) {
        std::istringstream iss(line);

        if (line[0] != '%') {
            if (line.substr(0, 14) == "%%MatrixMarket") {
                iss >> header >> object >> format >> field >> symmetry;
            } else {
                if (!parsed_first_line) {
                    iss >> ret.rows >> ret.cols >> ret.nnz;
                    parsed_first_line = true;
                } else {
                    int row, col;
                    dtype value;
                    iss >> row >> col >> value;
                    ret.elements[row][col] += value;
                }
            }
        } 
    }

    if (symmetry == "symmetric") {
        auto elements_copy = ret.elements;

        for (const auto& [row, col_value] : elements_copy) {
            for (const auto& [col, value] : col_value) {
                ret.elements[col][row] = value;    
            }
        }
    }

    return ret;
}