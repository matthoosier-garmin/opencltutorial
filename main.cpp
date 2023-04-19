#include <assert.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif

typedef float kernel_elem;
typedef std::vector<kernel_elem> kernel_row;
typedef std::vector<kernel_row> kernel_type;

kernel_elem gaussian (double x, double mu, double sigma)
{
	const double a = (x - mu) / sigma;
	return std::exp(-0.5 * a * a);
}

kernel_type produce2dGaussianKernel (unsigned int kernelRadius) {

	// Singularity if radius is zero. Just return back a 1x1 matrix with full weight
	// assigned to the single pixel.
	if (kernelRadius == 0) {
		return { { 1 } };
	}

	double sigma = kernelRadius / 2.;
	kernel_type kernel2d(2 * kernelRadius + 1, kernel_row(2 * kernelRadius + 1));
	double sum = 0;

	// compute values
	for (int row = 0; row < kernel2d.size(); row++) {
		for (int col = 0; col < kernel2d[row].size(); col++) {
			double x = gaussian(row, kernelRadius, sigma) * gaussian(col, kernelRadius, sigma);
			kernel2d[row][col] = x;
			sum += x;
		}
	}

	// normalize
	for (int row = 0; row < kernel2d.size(); row++) {
		for (int col = 0; col < kernel2d[row].size(); col++) {
			kernel2d[row][col] /= sum;
		}
	}

	return kernel2d;
}

struct Image
{
	std::vector<char> pixel;
	int width, height;
};

Image LoadImage (const char* path)
{
	std::ifstream in (path, std::ios::binary);

	std::string s;
	in >> s;

	if (s != "P6") {
		exit (1);
	}

	// Skip comments
	for (;;) {
		getline (in, s);

		if (s.empty ()) {
			continue;
		}

		if (s [0] != '#') {
			break;
		}
	}

	std::stringstream str (s);
	int width, height, maxColor;
	str >> width >> height;
	in >> maxColor;

	if (maxColor != 255) {
		exit (1);
	}

	{
		// Skip until end of line
		std::string tmp;
		getline(in, tmp);
	}

	std::vector<char> data (width * height * 3);
	in.read (reinterpret_cast<char*> (data.data ()), data.size ());

	const Image img = { data, width, height };
	return img;
}

void SaveImage (const Image& img, const char* path)
{
	std::ofstream out (path, std::ios::binary);

	out << "P6\n";
	out << img.width << " " << img.height << "\n";
	out << "255\n";
	out.write (img.pixel.data (), img.pixel.size ());
}

Image RGBtoRGBA (const Image& input)
{
	Image result;
	result.width = input.width;
	result.height = input.height;

	for (std::size_t i = 0; i < input.pixel.size (); i += 3) {
		result.pixel.push_back (input.pixel [i + 0]);
		result.pixel.push_back (input.pixel [i + 1]);
		result.pixel.push_back (input.pixel [i + 2]);
		result.pixel.push_back (0);
	}

	return result;
}

Image RGBAtoRGB (const Image& input)
{
	Image result;
	result.width = input.width;
	result.height = input.height;

	for (std::size_t i = 0; i < input.pixel.size (); i += 4) {
		result.pixel.push_back (input.pixel [i + 0]);
		result.pixel.push_back (input.pixel [i + 1]);
		result.pixel.push_back (input.pixel [i + 2]);
	}

	return result;
}

std::string GetPlatformName (cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

std::string GetDeviceName (cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetDeviceInfo (id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	return result;
}

cl_program CreateProgram (const std::string& source,
	cl_context context)
{
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateProgramWithSource.html
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}

int main (int argc, char* argv[])
{
	if (argc != 2) {
		printf("Usage: %s <filter size>\n", argv[0]);
		exit(1);
	}

	int filterSize = atoi(argv[1]);

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetPlatformIDs.html
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
		deviceIds.data (), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds [0]),
		0, 0
	};

	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext (contextProperties, deviceIdCount,
		deviceIds.data (), nullptr, nullptr, &error);
	CheckError (error);

	std::cout << "Context created" << std::endl;

	// Simple Gaussian blur filter
	float filter [2 * filterSize + 1][2 * filterSize + 1];

	kernel_type k = produce2dGaussianKernel(filterSize);

	assert((sizeof(filter) / sizeof(float)) == k[0].size() * k.size());

	for (int i = 0; i < k.size(); i++) {
		for (int j = 0; j < k[0].size(); j++) {
			filter[i][j] = k[i][j];
		}
	}

	// Create a program from source
	cl_program program = CreateProgram (LoadKernel ("kernels/image.cl"),
		context);

	char programOptions[256];
	snprintf(programOptions, sizeof(programOptions), "-D RADIUS=%d", filterSize);

	CheckError (clBuildProgram (program, deviceIdCount, deviceIds.data (), 
		programOptions, nullptr, nullptr));

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateKernel.html
	cl_kernel kernel = clCreateKernel (program, "Filter", &error);
	CheckError (error);

	// OpenCL only supports RGBA, so we need to convert here
	const auto image = RGBtoRGBA (LoadImage ("test.ppm"));

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateImage2D.html
	static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
	cl_mem inputImage = clCreateImage2D (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
		image.width, image.height, 0,
		// This is a bug in the spec
		const_cast<char*> (image.pixel.data ()),
		&error);
	CheckError (error);

	cl_mem outputImage = clCreateImage2D (context, CL_MEM_WRITE_ONLY, &format,
		image.width, image.height, 0,
		nullptr, &error);
	CheckError (error);

	// Create a buffer for the filter weights
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateBuffer.html
	cl_mem filterWeightsBuffer = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof (filter), filter, &error);
	CheckError (error);

	// Setup the kernel arguments
	clSetKernelArg (kernel, 0, sizeof (cl_mem), &inputImage);
	clSetKernelArg (kernel, 1, sizeof (cl_mem), &filterWeightsBuffer);
	clSetKernelArg (kernel, 2, sizeof (cl_mem), &outputImage);
	
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateCommandQueue.html
	cl_command_queue queue = clCreateCommandQueue (context, deviceIds [0],
		0, &error);
	CheckError (error);

	// Run the processing
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueNDRangeKernel.html
	std::size_t offset [3] = { 0 };
	std::size_t size [3] = { static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height), 1 };
	CheckError (clEnqueueNDRangeKernel (queue, kernel, 2, offset, size, nullptr,
		0, nullptr, nullptr));
	
	// Prepare the result image, set to black
	Image result = image;
	std::fill (result.pixel.begin (), result.pixel.end (), 0);

	// Get the result back to the host
	std::size_t origin [3] = { 0 };
	std::size_t region [3] = { static_cast<std::size_t>(result.width), static_cast<std::size_t>(result.height), 1 };
	clEnqueueReadImage (queue, outputImage, CL_TRUE,
		origin, region, 0, 0,
		result.pixel.data (), 0, nullptr, nullptr);

	SaveImage (RGBAtoRGB (result), "output.ppm");

	clReleaseMemObject (outputImage);
	clReleaseMemObject (filterWeightsBuffer);
	clReleaseMemObject (inputImage);

	clReleaseCommandQueue (queue);
	
	clReleaseKernel (kernel);
	clReleaseProgram (program);

	clReleaseContext (context);
}