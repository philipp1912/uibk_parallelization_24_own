#include "core/config.hpp"
#include "setup/fluid.hpp"
#include "setup/grid.hpp"
#include "solver/finite_volume_solver.hpp"
#include "solver/time_integrator.hpp"
#include "util/matrix.hpp"
#include "util/utility_functions.hpp"

#include "setup/mpi_handler.hpp"

#include <cmath>
#include <functional>
#include <iostream>

#include <mpi.h>


double Sedov_volume;

void init_Sedov(fluid_cell &fluid, double x_position, double y_position, double z_position) {
	double radius = sqrt(sim_util::square(x_position) + sim_util::square(y_position) + sim_util::square(z_position));
	double radius_init = 0.05;
	double volume_init = 4.0 / 3.0 * M_PI * radius_init * radius_init * radius_init;
	volume_init = Sedov_volume;
	double E_init = 1.0;
	double e_dens_init = E_init / volume_init;
	if (radius < 0.1) {
		fluid.fluid_data[fluid.get_index_energy()] = e_dens_init;
		fluid.fluid_data[fluid.get_index_tracer()] = 1.0;
	} else {
		fluid.fluid_data[fluid.get_index_energy()] = 1.e-5 / 0.4;
		fluid.fluid_data[fluid.get_index_tracer()] = 0.0;
	}
	fluid.fluid_data[fluid.get_index_density()] = 1.0;
	fluid.fluid_data[fluid.get_index_v_x()] = 0.0;
	fluid.fluid_data[fluid.get_index_v_y()] = 0.0;
	fluid.fluid_data[fluid.get_index_v_z()] = 0.0;
}

int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);

	std::vector<double> bound_low(3), bound_up(3);
	bound_low[0] = -0.5;
	bound_low[1] = -0.5;
	bound_low[2] = -0.5;

	bound_up[0] = 0.5;
	bound_up[1] = 0.5;
	bound_up[2] = 0.5;

	std::vector<int> num_cells(3);
	num_cells[0] = 64;
	num_cells[1] = 64;
	num_cells[2] = 64;

	const std::vector<int> v = {2, 2, 1};
	mpi_handler handler(v);

	grid_3D global_grid(bound_low, bound_up, num_cells, 2);
	grid_3D my_grid = handler.make_local_grid(global_grid);


	// Get number of Sedov cells
	double Sedov_volume_local = 0.0;
	int num_Sedov_cells_local = 0;
	double volume_cell = my_grid.x_grid.get_dx() * my_grid.y_grid.get_dx() * my_grid.z_grid.get_dx();

	for (int ix = 0; ix < my_grid.get_num_cells(0); ++ix) {
		double x_position = my_grid.x_grid.get_center(ix);
		for (int iy = 0; iy < my_grid.get_num_cells(1); ++iy) {
			double y_position = my_grid.y_grid.get_center(iy);
			for (int iz = 0; iz < my_grid.get_num_cells(2); ++iz) {
				double z_position = my_grid.z_grid.get_center(iz);
				double dist = sqrt(sim_util::square(x_position) + sim_util::square(y_position) + sim_util::square(z_position));
				if (dist < 0.1) {
					Sedov_volume_local += volume_cell;
					num_Sedov_cells_local++;
				}
			}
		}
	}


	double Sedov_volume_global;
	int num_Sedov_cells_global;

	MPI_Allreduce(&Sedov_volume_local, &Sedov_volume_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&num_Sedov_cells_local, &num_Sedov_cells_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	Sedov_volume = Sedov_volume_global * num_Sedov_cells_global;

	std::cout << " Volume of Sedov region: " << Sedov_volume_local << " in " << num_Sedov_cells_local << " cells\n";
	std::cout << " Sedov volume total: " << 1.0/Sedov_volume << "\n";

	// Now, I will create a HD fluid
	fluid hd_fluid(parallelisation::FluidType::adiabatic);
	hd_fluid.setup(my_grid);

	std::function<void(fluid_cell &, double, double, double)> function_init = init_Sedov;

	finite_volume_solver solver(hd_fluid, handler, global_grid);
	solver.set_init_function(function_init);

	MPI_Finalize();

	return 0;

	double t_final = 0.1;
	double dt_out = 0.005;

	solver.run(my_grid, hd_fluid, t_final, dt_out);

	return 0;
}