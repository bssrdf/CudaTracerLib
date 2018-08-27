//This is an example renderer using the library.
//All that is done here is using the Mitsuba loader to load a scene and render a number of passes with a specified integrator.
#include <StdAfx.h>
#include <iostream>
#include <algorithm>
#include <filesystem.h>
#include <cctype>
#include <Engine/Core.h>
#include <Engine/DynamicScene.h>
#include <SceneTypes/Node.h>
#include <Engine/Material.h>
#include <Base/Buffer.h>
#include <SceneTypes/Light.h>
#include <Engine/Image.h>
#include <Integrators/PrimTracer.h>
#include <Integrators/PathTracer.h>
#include <Integrators/PhotonTracer.h>
#include <Integrators/Bidirectional/BDPT.h>
#include <Integrators/ProgressivePhotonMapping/PPPMTracer.h>
#include <Integrators/PseudoRealtime/WavefrontPathTracer.h>
#include <Kernel/ImagePipeline/ImagePipeline.h>
#include <Engine/SceneLoader/Mitsuba/MitsubaLoader.h>


#ifdef _WIN32
#include <windows.h>
#include <GL/glew.h>
#include <sstream>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>
#include <GL/gl.h>
#endif

#include <GL/freeglut.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
//https://stackoverflow.com/questions/33990387/mvs2015-keeps-giving-unresolved-externals-error-with-sdl2
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

using namespace CudaTracerLib;

SDL_Window *sdl_window;

int width = 1024;
int height = 1024;

// The CUDA kernels will draw directly inside an OpenGL buffer (PBO)
// which will be displayed via an OpenGL texture.

// OpenGL buffer and texture helper functions

void create_buffer(GLuint* buffer)
{
	glGenBuffersARB(1, buffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, *buffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, width*height*4, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void destroy_buffer(GLuint* buffer)
{
	glBindBuffer(GL_TEXTURE_2D, 0);
	glDeleteBuffers(1, buffer);
	*buffer = 0;
}

void create_texture(GLuint* tex)
{
	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_2D, *tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void destroy_texture(GLuint* tex)
{
	glBindTexture(GL_TEXTURE_2D, *tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDeleteTextures(1, tex);
}


class SimpleFileManager : public IFileManager
{
    std::string data_path;
public:
    SimpleFileManager(const std::string& p)
        :data_path(p)
    {

    }
    virtual std::string getCompiledMeshPath(const std::string& name)
    {
        return data_path + "Compiled/" + name;
    }
    virtual std::string getTexturePath(const std::string& name)
    {
        return data_path + "textures/" + name;
    }
    virtual std::string getCompiledTexturePath(const std::string& name)
    {
        return data_path + "Compiled/" + name;
    }
};

//quick and dirty argument parsing since boost program options is a compiled library
struct options
{
    std::string data_path;
    std::string scene_file;
    int n_passes;
    TracerBase* tracer;
};
std::optional<options> parse_arguments(int ac, char** av)
{
    options opt;

    auto is_number = [](const std::string& s)
    {
        return !s.empty() && std::find_if(s.begin(),
            s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
    };

    std::vector<std::string> tracers = { "direct", "PT", "PT_Wave", "BDPT", "PPPM" };

    auto print_error = [&](const std::string& arg)
    {
        //print error message

        std::cout << "accepts 4 arguments : data path, scene file path, number of passes and tracer type {";
        for (auto& t : tracers)
            std::cout << t << ", ";
        std::cout << "}" << std::endl << arg << " could not be used, exiting now" << std::endl;
    };

    int n_args_used = 0;
    for (int i = 1; i < ac; i++)
    {
        std::string arg = av[i];
        if (std::filesystem::is_directory(std::filesystem::path(arg)))
            opt.data_path = arg + "/";
        else if (std::filesystem::is_regular_file(std::filesystem::path(arg)))
            opt.scene_file = arg;
        else if (is_number(arg))
            opt.n_passes = std::stoi(arg);
        else if (std::find(tracers.begin(), tracers.end(), arg) != tracers.end())
        {
            if (arg == "direct")
                opt.tracer = new PrimTracer();
            else if (arg == "PT")
                opt.tracer = new PathTracer();
            else if (arg == "PT_Wave")
                opt.tracer = new WavefrontPathTracer();
            else if (arg == "BDPT")
                opt.tracer = new BDPT();
            else if (arg == "PPPM")
                opt.tracer = new PPPMTracer();
        }
        else
        {
            print_error(arg);

			return {};
        }

        n_args_used++;
    }

    if (n_args_used != 4)
    {
        print_error("");
		return {};
    }

    return opt;
}


//https://stackoverflow.com/a/36315819
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage)
{
	int val = (int)(percentage * 100);
	int lpad = (int)(percentage * PBWIDTH);
	int rpad = PBWIDTH - lpad;
	printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
	fflush(stdout);
}

int main(int ac, char** av)
{
    auto opt_options = parse_arguments(ac, av);
    if (!opt_options)
        return 1;
    auto options = opt_options.value();


    //int width = 1024, height = 1024;
    const float fov = 90;
    InitializeCuda4Tracer(options.data_path);
    SimpleFileManager fManager = { options.data_path };
    Sensor camera = CreateAggregate<Sensor>(PerspectiveSensor(width, height, fov));
    DynamicScene scene(&camera, SceneInitData::CreateForScene(1000, 30000, 100), &fManager);

    std::optional<Vec2i> img_size;
    ParseMitsubaScene(scene, options.scene_file, std::map<std::string, std::string>(), img_size, false, false, false);
    if (img_size)
    {
        width = img_size.value().x;
        height = img_size.value().y;
    }

	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cout << "SDL_Init Error: " << SDL_GetError() << std::endl;
		return 1;
	}

	sdl_window = SDL_CreateWindow(
		"An SDL2 window",                  // window title
		SDL_WINDOWPOS_UNDEFINED,           // initial x position
		SDL_WINDOWPOS_UNDEFINED,           // initial y position
		width,                               // width, in pixels
		height,                               // height, in pixels
		SDL_WINDOW_OPENGL                  // flags - see below
	);
		
	if (sdl_window == NULL){
		std::cout << "Couldn't set video mode: " << SDL_GetError() << std::endl;
		return 1;
	}		

	// Create an OpenGL context associated with the window.
	SDL_GLContext glcontext = SDL_GL_CreateContext(sdl_window);
	
    Image outImage(width, height);

    options.tracer->Resize(width, height);
    options.tracer->InitializeScene(&scene);
    scene.UpdateScene();

	options.tracer->Debug(&outImage, Vec2i(132, 472));

    for (int i = 0; i < options.n_passes; i++)
    {
        options.tracer->DoPass(&outImage, !i);
		printProgress(i / double(options.n_passes));
    }

    applyImagePipeline(*options.tracer, outImage, CreateAggregate<Filter>(BoxFilter(0.5f, 0.5f)));

    outImage.WriteDisplayImage("result.png");

    outImage.Free();
    DeInitializeCuda4Tracer();

	// Once finished with OpenGL functions, the SDL_GLContext can be deleted.
	SDL_GL_DeleteContext(glcontext);

	SDL_DestroyWindow(sdl_window);

	// Clean up
	SDL_Quit();


    return 0;
}
