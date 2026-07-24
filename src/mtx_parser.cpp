#include "mtx_parser.hpp"

#include <fstream>
#include <sstream>
#include <map>

MtxParser::COOEntry::COOEntry(int row, int col, dtype value) :
    row(row), col(col), value(value) {}

bool operator<(const MtxParser::COOEntry& lhs, const MtxParser::COOEntry& rhs) {
    if (lhs.row != rhs.row) {
        return lhs.row < rhs.row;
    }

    return lhs.col < rhs.col;
}

MtxParser::MtxMatrix MtxParser::parseMtxFile(const std::string& file_path) {
    MtxParser::MtxMatrix ret;
    std::map<int, std::map<int, int>> elements; 
    std::string header, object, format, field, symmetry;

    std::ifstream ifs(file_path);

    if (ifs.fail()) {
        std::cerr << "Unable to open file: " << file_path << std::endl;
        return ret;
    }

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
                    elements[row - 1][col - 1] = value;
                }
            }
        } 
    }

    if (symmetry == "symmetric") {
        auto elements_copy = elements;

        for (const auto& [row, col_value] : elements_copy) {
            for (const auto& [col, value] : col_value) {
                elements[col][row] = value;    
            }
        }
    }

    for (const auto& [row, col_value] : elements) {
        for (const auto& [col, value] : col_value) {
            ret.entries.emplace_back(row, col, elements[col][row]);
        }
    }

    return ret;
}