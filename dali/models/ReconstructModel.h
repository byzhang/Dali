#ifndef RECONSTRUCT_MODEL_MAT_H
#define RECONSTRUCT_MODEL_MAT_H

#include <vector>
#include <string>
#include <sstream>
#include "dali/utils.h"

namespace Indexing {
    class Index;
}

class ReconstructModel {
    public:
        virtual std::vector<int> reconstruct(
            Indexing::Index,
            int,
            int symbol_offset = 0) const = 0;

        virtual std::vector<utils::OntologyBranch::shared_branch> reconstruct_lattice(
            Indexing::Index,
            utils::OntologyBranch::shared_branch,
            int) const = 0;

        std::string reconstruct_string(
            Indexing::Index,
            const utils::Vocab&,
            int,
            int symbol_offset = 0) const;
        std::string reconstruct_lattice_string(
            Indexing::Index,
            utils::OntologyBranch::shared_branch,
            int) const;
};

#endif
