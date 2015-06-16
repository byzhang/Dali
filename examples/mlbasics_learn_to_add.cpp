#include <cstring>
#include <gflags/gflags.h>
#include <memory>

#include "dali/core.h"

typedef Mat<double> mat;


int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
    "RNN Kindergarden - Lesson 1 - Learning to add.");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    dali_init();

    // How many examples to put in dataset
    const int NUM_EXAMPLES = 100;
    // How many number to add.
    const int EXAMPLE_SIZE = 3;
    // How many iterations of gradient descent to run.
    const int ITERATIONS = 150;
    // What is the learning rate.
    double LR = 0.01;

    // Generate random examples, all the rows sum to number between 0 and 1.
    mat X(NUM_EXAMPLES, EXAMPLE_SIZE, weights<double>::uniform(0.0, 1.0/EXAMPLE_SIZE));

    // Compute sums of elements for each example. This is what we would
    // like the network to output.
    // TODO(szymon): this should really be X.rowwise().sum()
    auto ones = MatOps<double>::fill(mat(X.dims(1),1), 1);
    auto Y = X.dot(ones);

    // this is our dataset - please do not
    // consider during backpropagation.
    X = MatOps<double>::consider_constant(X);
    Y = MatOps<double>::consider_constant(Y);
    // Those are our parameters: y_output = W1*X1 + ... + Wn*Xn
    // We initialize them to random numbers between 0 and 1.
    mat W(EXAMPLE_SIZE, 1, weights<double>::uniform(1.0));

    W.print();

    for (int i = 0; i < ITERATIONS; ++i) {
        // What the network predicts the output will be.
        mat predY = X.dot(W);
        // Squared error between desired and actual output
        // E = sum((Ypred-Y)^2)
        mat error = ( (predY - Y) ^ 2 ).sum();
        // Mark error as what we compute error with respect to.
        error.grad();
        // Print error so that we know our progress.
        error.print();
        // Perform backpropagation algorithm.
        graph::backward();
        // Use gradient descent to update network parameters.
        // This is slightly obnoxious, but fear not - we
        // provide an excellent solver class, so that you
        // never how to do it on your own!
        if (W.w()->w.prefers_cpu()) {
            auto www = W.w()->w.mutable_cpu_data();
            auto wwwddd = W.dw()->dw.cpu_data();
            www -= LR * wwwddd;
        } else {
            std::cout << "no sorry" << std::endl;
            // need to wrap it in matops sgd step
            // or compile as CU
            // auto www = W.w()->w.mutable_gpu_data();
            // auto wwwddd = W.dw()->dw.mutable_gpu_data();
            // www -= LR * wwwddd;
            // std::cout << W.w()->w.gpu_fresh << std::endl;
        }
        // Reset gradients
        W.clear_grad();
    }
    // Print the weights after we are done. The should all be close to one.
    W.print();
}
