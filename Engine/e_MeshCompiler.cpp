#include "StdAfx.h"
#include "e_MeshCompiler.h"
#include <algorithm>
#include <string>
#include "..\Base\StringUtils.h"

bool hasEnding (std::string const &fullString, std::string const &_ending)
{
	std::string ending = _ending;
	toLower(ending);
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

bool e_ObjCompiler::IsApplicable(const char* a_InputFile, IInStream& in, e_MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".obj");
	if(out && b)
		*out = e_MeshCompileType::Static;
	return b;
}

void e_ObjCompiler::Compile(IInStream& in, OutputStream& a_Out)
{
	compileobj(in, a_Out);
}

bool e_Md5Compiler::IsApplicable(const char* a_InputFile, IInStream& in, e_MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".md5mesh");
	if(out && b)
		*out = e_MeshCompileType::Animated;
	return b;
}

#if defined(ISWINDOWS)
#include <Windows.h>
void e_Md5Compiler::Compile(IInStream& in, OutputStream& a_Out)
{
	std::vector<std::string> A;
	char dir[255];
	ZeroMemory(dir, sizeof(dir));
	char drive[255];
	ZeroMemory(drive, sizeof(drive));
	_splitpath(in.getFilePath(), drive, dir, 0, 0);
	WIN32_FIND_DATA dat;
	HANDLE hFind = FindFirstFile((std::string(drive) + std::string(dir) + "\\*.md5anim").c_str(), &dat);
	while(hFind != INVALID_HANDLE_VALUE)
	{
		A.push_back(std::string(drive) + std::string(dir) + std::string(dat.cFileName));
		if(!FindNextFile(hFind, &dat))
			break;
	}
	e_AnimatedMesh::CompileToBinary(in.getFilePath(), A, a_Out);
	FindClose(hFind);
}
#elif defined(ISUNIX)
#include <dirent.h>
void e_Md5Compiler::Compile(const char* a_InputFile, OutputStream& a_Out)
{
	c_StringArray A;
	char dir[255];
	ZeroMemory(dir, sizeof(dir));
	_splitpath(a_InputFile, 0, dir, 0, 0);
	struct dirent **namelist;
	int n;
    n = scandir(".*md5anim", &namelist, 0, alphasort); 
	while(n--)
	{ 
        A(std::string(dir) + std::string(namelist[n]->d_name)); 
        free(namelist[n]); 
    }
	e_AnimatedMesh::CompileToBinary(a_InputFile, A, a_Out);
}
#endif



bool e_PlyCompiler::IsApplicable(const char* a_InputFile, IInStream& in, e_MeshCompileType* out)
{
	bool b = hasEnding(a_InputFile, ".ply");
	if(b)
	{
		char magic[9];
		magic[8] = 0;
		in.Read(magic, 8);
		b = std::string(magic) != "EPLYBNDS";
		in.Move(-8);
	}
	if(out && b)
		*out = e_MeshCompileType::Static;
	return b;
}

void e_PlyCompiler::Compile(IInStream& in, OutputStream& a_Out)
{
	compileply(in, a_Out);
}

void e_MeshCompilerManager::Compile(IInStream& in, const char* a_InputFile, OutputStream& a_Out, e_MeshCompileType* out)
{
	e_MeshCompileType t;
	for(unsigned int i = 0; i < m_sCompilers.size(); i++)
	{
		bool b = m_sCompilers[i]->IsApplicable(a_InputFile, in, &t);
		if(b)
		{
			if(out)
				*out = t;
			a_Out << (unsigned int)t;
			m_sCompilers[i]->Compile(in, a_Out);
			break;
		}
	}
}