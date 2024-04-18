#include "solver/limiter.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cmath>

int signum(double x) {
	return (x > 0) - (x < 0);
}

limiter_base::limiter_base() {}

limiter_minmod::limiter_minmod(double theta) { this->theta = theta; }

double limiter_minmod::compute(double first, double second, double third) {
	// TBD by students
	double abs_first = std::abs(first);
	double abs_second = std::abs(second);
	double abs_third = std::abs(third);
	double min_value = std::min({abs_first, abs_second, abs_third});

	if (signum(first) == signum(second) && signum(second) == signum(third))
		return signum(first) * min_value;
	else
		return 0.0;
}