// Testbench for SW/HW CNN implementations

#include "sqj2_tb.hpp"


int main( int argc, char* argv[] )
{

	// Read arguments
	if ( ( argc != 2 ) )
	{
		DEBUG( "ERROR: " << std::endl
		<< "Forgot to provide the error threshold" << std::endl
		<< "for checking the C/C++ implementation results" << std::endl
		<< "against the Caffe/Ristretto implementation results." << std::endl
		<< "e.g. ./conv_tb 0.0001" << std::endl
		<< "Exiting!");
		return -1;
	}
	float  error_sz = atof( argv[1] );

	// Uncomment to test the desired implementation

	// Test SqueezeNet v1.1 8-bit dynamic fixed-point implementation
	return test_sqn_dfp(error_sz);

	// Test SqueezeNet v1.1 fixed-point implementation
	//return test_sqn_fip(error_sz);

	// Test SqueezeNet v1.1 fixed-point separated HW accelerator implementation
	//return test_sqn_fips(error_sz);

	// Test ZynqNet 8-bit dynamic fixed-point implementation
	//return test_zqn_dfp(error_sz);

	// Test ZynqNet floating-point implementation
	//return test_zqn_flp(error_sz);

}
