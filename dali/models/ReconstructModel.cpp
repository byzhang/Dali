#include "dali/models/ReconstructModel.h"
#include "dali/mat/Index.h"

using std::string;
using std::stringstream;

string ReconstructModel::reconstruct_string(
        Indexing::Index example,
        const utils::Vocab& lookup_table,
        int eval_steps,
        int symbol_offset) const {
    auto reconstruction = reconstruct(example, eval_steps, symbol_offset);
    stringstream rec;
    for (auto& cat : reconstruction) {
        rec << (
            (cat < lookup_table.size()) ?
                lookup_table.index2word.at(cat) :
                (
                    cat == lookup_table.size() ? utils::end_symbol : "??"
                )
            ) << ", ";
    }
    return rec.str();
}

string ReconstructModel::reconstruct_lattice_string(
        Indexing::Index example,
        utils::OntologyBranch::shared_branch root,
        int eval_steps) const {
    auto reconstruction = reconstruct_lattice(example, root, eval_steps);
    stringstream rec;
    for (auto& cat : reconstruction)
        rec << ((&(*cat) == &(*root)) ? "⟲" : cat->name) << ", ";
    return rec.str();
}
