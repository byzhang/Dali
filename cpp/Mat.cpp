#include "Mat.h"
#include "utils.h"

using namespace Eigen;

template<typename T>
Mat<T>::Mat (int _n, int _d) : n(_n), d(_d), random_id(utils::get_random_id()) {
    this->w = eigen_mat::Zero(n,d);
    this->dw = eigen_mat::Zero(n,d);
}
template<typename T>
Mat<T>::Mat (int _n, int _d, bool empty) : n(_n), d(_d), random_id(utils::get_random_id()) {
	this->w  = empty ? eigen_mat(n,d) : eigen_mat::Zero(n,d);
	this->dw = eigen_mat::Zero(n,d);
}

template<typename T>
void Mat<T>::print () {
	for (int i = 0; i < n ; ++i) {
		std::cout << (i == 0 ? "[" : " ");
		for (int j = 0; j < d; ++j) {
			std::cout << std::fixed
			          << std::setw( 7 ) // keep 7 digits
			          << std::setprecision( 3 ) // use 3 decimals
			          << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
			          << this->w(i,j) << " ";
		}
		std::cout << (i == n-1 ? "]" : "\n");
	}
	std::cout << std::endl;
}

template<typename T>
Mat<T>::~Mat() {}

template<typename T>
Mat<T>::Mat (int _n, int _d, T std) : n(_n), d(_d), random_id(utils::get_random_id()) {
	std::default_random_engine generator;
	std::normal_distribution<T> distribution(0.0, std);
	std::random_device rd;
	generator.seed(rd());
	auto randn = [&] (int) {return distribution(generator);};
	this->w = eigen_mat::NullaryExpr(n,d, randn);
	this->dw = eigen_mat(n,d);
}

template<typename T>
Mat<T> Mat<T>::RandMat(int n, int d, T std) {
	// is in fact using C++ 11 's rvalue, move operator,
	// so no copy is made.
	return Mat(n, d, std);
}

template<typename T>
Mat<T> Mat<T>::Empty(int n, int d) {
	// use an empty matrix and modify
	// it so as to not incur the filling
	// with zeros cost.
	return Mat(n, d, true);
}

template<typename T>
std::ostream& operator<<(std::ostream& strm, const Mat<T>& a) {
	return strm << "<#Mat n=" << a.n << ", d=" << a.d << ">";
}

template <typename T>
std::size_t std::hash<Mat<T>>::operator()(const Mat<T>& k) const {
	return k.random_id;
}

template <typename T>
static bool operator!=(const Mat<T>& matrix1, const Mat<T>& matrix2) {
    return matrix2.random_id != matrix1.random_id;
}

template <typename T>
static bool operator==(const Mat<T>& matrix1, const Mat<T>& matrix2) {

    return matrix2.random_id == matrix1.random_id;
}

template<typename T> Backward<T>::Backward (
	shared_mat matrix1,
	shared_mat out,
	uint type) {
	this->matrix1 = matrix1;
	this->matrix2 = NULL;
	this->out = out;
	this->type = type;
}

template<typename T> Backward<T>::Backward (
	shared_mat matrix1,
	shared_mat out,
	int index,
	uint type) {
	this->matrix1 = matrix1;
	this->matrix2 = NULL;
	this->out = out;
	this->type = type;
	this->ix = index;
}
template<typename T> Backward<T>::Backward (
	shared_mat matrix1,
	shared_mat matrix2,
	shared_mat out,
	uint type) {
	this->matrix1 = matrix1;
	this->matrix2 = matrix2;
	this->out = out;
	this->type = type;
}

template<typename T>
std::ostream& operator<<(std::ostream& strm, const Backward<T>& a) {
	if (a.matrix2 != NULL) {
		return strm << "<#Backward matrix1=" << *(a.matrix1) << ", matrix2=" << *(a.matrix2) << ", out=" << *(a.out) << ", type=\""<< a.op_type() << "\">";
	}
	return strm << "<#Backward matrix1=" << *(a.matrix1) << ", out=" << *(a.out) << ", type=\""<< a.op_type() << "\">";
}

template std::ostream& operator<< <double>(std::ostream& strm, const Backward<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const Backward<float>& a);

template<typename T> 
std::string Backward<T>::op_type () const {
	switch(this->type) {
		case utils::ops::add:
			return "add";
			break;
		case utils::ops::eltmul:
			return "eltmul";
			break;
		case utils::ops::tanh:
			return "tanh";
			break;
		case utils::ops::sigmoid:
			return "sigmoid";
			break;
		case utils::ops::relu:
			return "relu";
			break;
		case utils::ops::mul:
			return "mul";
			break;
		case utils::ops::row_pluck:
			return "row_pluck";
			break;
		case utils::ops::add_broadcast:
			return "add_broadcast";
			break;
		case utils::ops::eltmul_broadcast:
			return "eltmul_broadcast";
			break;
		default:
			return "?";
			break;
	}
}

template<typename T> 
void Backward<T>::operator ()() {
	switch(this->type) {
		case utils::ops::add:
			this->matrix1->dw += this->out->dw;
			this->matrix2->dw += this->out->dw;
			break;
		case utils::ops::add_broadcast:
			this->matrix1->dw += this->out->dw;
			this->matrix2->dw += this->out->dw.rowwise().sum();
			break;
		case utils::ops::eltmul:
			this->matrix1->dw += ((this->matrix2->w).array() * (this->out->dw).array()).matrix();
			this->matrix2->dw += ((this->matrix1->w).array() * (this->out->dw).array()).matrix();
			break;
		case utils::ops::eltmul_broadcast:
			this->matrix1->dw += ((this->out->dw).array().colwise() * (this->matrix2->w).col(0).array()).matrix();
			this->matrix2->dw += ((this->matrix1->w).array() * (this->out->dw).array()).matrix().rowwise().sum();
			break;
		case utils::ops::sigmoid:
			this->matrix1->dw += (((this->out->w).array() * (1.0 - this->out->w.array())) * this->out->dw.array()).matrix();
			break;
		case utils::ops::mul:
			this->matrix1->dw += (this->out->dw) * ((this->matrix2->w).transpose());
			this->matrix2->dw += this->matrix1->w.transpose() * (this->out->dw);
			break;
		case utils::ops::relu:
			this->matrix1->dw += (this->out->w.unaryExpr(utils::sign_operator<T>()).array() * this->out->dw.array()).matrix();
			break;
		case utils::ops::tanh:
			this->matrix1->dw += (this->out->w.unaryExpr(utils::dtanh_operator<T>()).array() * this->out->dw.array()).matrix();
			break;
		case utils::ops::row_pluck:
			this->matrix1->dw.row(this->ix) += (this->out->w.array() * this->out->dw.array()).matrix().col(0).transpose();
			break;
		default:
			std::stringstream error_msg;
			error_msg << "NotImplemented: Do not know how to backpropagate for this type => "
			   << op_type() << " (" << this->type << ")";
			throw std::invalid_argument(error_msg.str());
			break;
	}
}

template<typename T>
std::shared_ptr<Mat<T>> softmax(std::shared_ptr<Mat<T>> matrix) {
	T layer_max = matrix->w.array().maxCoeff();
	Array<T, Dynamic, Dynamic> exped_distributions = (matrix->w.array() - layer_max).exp();
	T total_distribution = exped_distributions.sum();
	auto out = std::make_shared<Mat<T>>(
		matrix->n,
		matrix->d,
		true);
	out->w = (exped_distributions / total_distribution).matrix();
	return out;
}

template<typename T>
Graph<T>::Graph (bool _needs_backprop) : needs_backprop(_needs_backprop) {}
template<typename T>
Graph<T>::Graph () : needs_backprop(true) {}

template<typename T>
void Graph<T>::backward () {
	std::cout << "Graph backprop vector contains (in reverse):\n";
	for (auto it = this->backprop.rbegin(); it != this->backprop.rend(); ++it) {
		std::cout << ' ' << *it << "\n";
		(*it)();
	}
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul_broadcast(
	shared_mat matrix1,
	shared_mat matrix2) {
	if (matrix1->n != matrix2->n || matrix2->d != 1)
		throw std::invalid_argument("Matrices cannot be element multiplied with broadcast, they do not have the same dimensions.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.array().colwise() * matrix2->w.col(0).array()).matrix();
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::eltmul_broadcast);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul(
	shared_mat matrix1,
	shared_mat matrix2) {
	if (matrix1->d != matrix2->d && (matrix1->d == 1 || matrix2->d == 1)) {
		if (matrix1->d == 1) {
			return eltmul_broadcast(matrix2, matrix1);
		}
		return eltmul_broadcast(matrix1, matrix2);
	}
	if (matrix1->n != matrix2->n || matrix1->d != matrix2->d)
		throw std::invalid_argument("Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.array() * matrix2->w.array()).matrix();
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::eltmul);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add_broadcast(shared_mat matrix1, shared_mat matrix2) {
	// broadcast matrix 2:
	if (matrix1->n != matrix2->n || matrix2->d != 1)
		throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
	auto out = std::make_shared<Mat<T>>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = (matrix1->w.colwise() + matrix2->w.col(0)).matrix();
	if (this->needs_backprop)
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::add_broadcast);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add(
		shared_mat matrix1,
		shared_mat matrix2) {
	if (matrix1->d != matrix2->d && (matrix1->d == 1 || matrix2->d == 1)) {
		if (matrix1->d == 1) {
			return add_broadcast(matrix2, matrix1);
		}
		return add_broadcast(matrix1, matrix2);
	}
	if (matrix1->n != matrix2->n || matrix1->d != matrix2->d)
		throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
	auto out = std::make_shared<Mat<T>>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w + matrix2->w;
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::add);
	return out;
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::sigmoid(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::sigmoid_operator<T>());
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, utils::ops::sigmoid);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::tanh(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::tanh_operator<T>());
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, utils::ops::tanh);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::relu(shared_mat matrix1) {
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix1->d,
		true);
	out->w = matrix1->w.unaryExpr(utils::relu_operator<T>());
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, utils::ops::relu);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul(
	shared_mat matrix1,
	shared_mat matrix2) {
	if (matrix1->d != matrix2->n)
		throw std::invalid_argument("matmul dimensions misaligned.");
	auto out = std::make_shared<mat>(
		matrix1->n,
		matrix2->d,
		true);
	out->w = matrix1->w * matrix2->w;
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, matrix2, out, utils::ops::mul);
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::rows_pluck(
	shared_mat matrix1,
	std::vector<int> indices
	) {
	auto out = std::make_shared<mat>(
		matrix1->d,
		indices.size(),
		true);
	int offset = 0;
	for (auto& i : indices) {
		out->w.col(offset) = matrix1->w.row(i).transpose();
		++offset;
	}
	return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::row_pluck(
	shared_mat matrix1,
	int ix) {
	auto out = std::make_shared<mat>(
		matrix1->d,
		1,
		true);
	out->w = matrix1->w.row(ix).transpose();
	if (this->needs_backprop)
		// allocates a new backward element in the vector using these arguments:
		this->backprop.emplace_back(matrix1, out, ix, utils::ops::row_pluck);
	return out;
}

template<typename T>
Solver<T>::Solver (
            T _decay_rate,
            T _smooth_eps,
            T _clipval) :
        decay_rate(_decay_rate),
        smooth_eps(_smooth_eps),
        clipval(_clipval),
        clip_values(-_clipval, _clipval) {};

template<typename T>
Solver<T>::~Solver () {};

template<typename T>
void Solver<T>::step(
	std::vector<shared_mat>& model,
	T step_size,
	T regc
	) {
	for (auto& param : model) {
		// this operation should be run once unless
		// we expect the parameters of the model
		// to change online (probably not the case)
		if (!(this->step_cache.count(*param) > 0)) {
			auto new_cache = this->step_cache.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(*param),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);
		}
		auto& s = this->step_cache[*param];
		// update gradient cache using decay rule:
		s = s * this->decay_rate + (1.0 - this->decay_rate) * param->dw.unaryExpr(this->square_values);
		// clip the gradient to prevent explosions:
		param->dw = (param->dw).array().unaryExpr(this->clip_values).matrix();
		// update gradient using RMSprop rule
		param->w += step_size * (param->dw.array() / (s.array() + this->smooth_eps).sqrt() ).matrix()  - (regc * param->w);
		// reset gradient
		param->dw.fill(0);
	}
}
template class Mat<float>;
template class Mat<double>;

template class Backward<float>;
template class Backward<double>;

template class Graph<float>;
template class Graph<double>;

template class Solver<float>;
template class Solver<double>;