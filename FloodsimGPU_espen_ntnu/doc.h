/**
\mainpage
\author André R Brodtkorb, <Andre.Brodtkorb@sintef.no>
\author Martin L Sætra, <Martin.Lilleeng.Saetra@sintef.no>

\section platforms Tested Platforms

\section installation Installation for developers on Windows
These instructions are for developers of KPSimulator. Most of these instructions are
for windows users. If you are using linux, use your package manager to install the
prerequisites. Compiling should go smoothly.

\subsection prerequisites Prerequisites
Try installing the version of the prerequisites as shown in tested platforms.
Other versions might work, but have not been tested.

\subsubsection boost Boost
Boost is a set of libraries and header files that are actively developed, and of 
extremely high quality. Boost includes utilities for cross-platform development of
e.g., threaded code. C++ developers should make themselves familiar with Boost.

-# Go to http://boost.org, and download.
-# Unzip the contents to of the downloaded file to a suitable location (e.g., your desktop)
-# Now compile and install boost:
	-# Go to http://boost.org, and click [getting started], followed by [getting started on windows].
	-# Click on menu item "5.3 Or, build binaries from source"
	-# Download bjam, and place bjam.exe in the boost directory you just created (on your desktop)
	-# Open a command line (cmd), and go to the boost directory (where you just placed bjam)
	-# Compile and install boost by typing
	For 32-bit systems:
	\verbatim
	bjam --build-dir=build --build-type=complete --with-thread --with-date_time --with-filesystem 
		--with-system --with-program_options --with-test install
	\endverbatim
	For 64-bit systems:
	\verbatim
	bjam address-model=64 --build-dir=build --build-type=complete --with-thread --with-date_time 
		--with-filesystem --with-system --with-program_options --with-test install
	\endverbatim
	This should build boost, and place the libraries and header files in C:\Boost\. This takes a couple
	of minutes, so you can safely go and have a cup of coffee. If you do not drink coffee, stop reading
	now. You will never be able to complete the rest of these instructions.

\subsubsection cmake CMake
CMake is a cross-platform make-file generator. It takes as input a set of cmake-scripts, that describe
how source code should be built. From these cmake-scripts, CMake can generate build files for linux 
(Makefiles), Windows (using MSVC project and solution files), and probably also mac, if you would ever
want to chuck your soul into the Apple-abyss.

-# Go to http://cmake.org
-# Download and install cmake 2.8.8 or newer.

\subsubsection cuda NVIDIA CUDA
CUDA is a programming language from NVIDIA used to program their graphics processing units
for general purpose computation. It consists of three parts: the toolkit, used to compile 
CUDA source code into binary programs; the driver, that works as an operating system for your
graphics card and enables you to run programs on it; and the SDK, which contains sample code, etc.

-# Go to http://nvidia.com/cuda
-# Go to developer downloads, download, and install the three different parts of CUDA: 
	driver, toolkit, and SDK.
	
\subsubsection MSVC Redistributables
The visual studio compiler contains a set of DLLs that are called by programs created with
visual studio. These include the C runtime, OpenMP runtime, etc. When you install Visual
studio, sometimes its DLLs are not added to the system search path, and you will get errors
when you launch programs. This can be fixed in several ways:

-# Do not run the programs form the command line. Run them from the Visual Studio command line,
	e.g., run "Visual Studio 2008 Command Prompt" from your start menu instead of running
	"cmd" or "Command Prompt"
-# Install a MSVC Redist package. Search microsoft.com for the appropriate redistribution package
	for your compiler. For 32 bit MSVC 9.0, it is called "Microsoft Visual C++ 2008 Redistributable 
	Package (x86)"
	
Note that the second option is required for users of pre-compiled binaries, unless you supply them
with an installer that includes them.

\subsection building Building the source code
After having installed all of the above mentioned requirements, we are ready to configure
and run the build.

-#Unzip the source code to a local directory. The following text assumes it has been unzipped to
\verbatim
	CUDASim
\endverbatim
	on your desktop.

You should now be able to generate the build files:
-# Open cmake-gui from your start menu, and point the source to CUDASim
	and the build to CUDASim\build
-# Select configure, and choose your native compiler from the drop-down list. Make sure
	that you choose the right 32/64-bit value here as well.
-# Select grouped view, and set the CUDA and Boost dirs appropriately.
-# Select configure, and hopefully, you will not have any errors.
-# Finally select generate.

We have now created the project files for visual studio. Go to 
\verbatim
CUDASim\build\
\endverbatim
and open the solution file by double-clicking it. After opening it, simply select build->build all
and the simulator should build without problems. If you do have problems, check that you have given
CMake the correct paths, etc.

\subsection running Running the examples
The first example we will use is the dam break over triangular bump case, as defined by the EU CADAM
project. Open a command line (make sure that it is a Visual Studio Command Prompt, or that you have
installed the redistributable package mentioned in prerequisites), and go to 
\verbatim
CUDASim\build\bin
\endverbatim
Proceed in to Debug or Release, depending on what type of build you have chosen. You should now see
\verbatim
kp_eu_cadam.exe
\endverbatim
among the files listed. This test application takes as input number of grid cells, and simulates
90 seconds of the dam break. Try running
\verbatim
kp_eu_cadam.exe 760 > eu_cadam.csv
\endverbatim
and then open eu_cadam.csv in Microsoft Excel. Try plotting the time variable against any of the gauges.

Let us now try to run some other real-world simulations. Go to 
\verbatim
CUDASim\data\malpasset
\endverbatim
Malpasset was a dam in south-eastern France that broke and over 400 persons died. Try running
\verbatim
..\..\KPSimulatorTest\build\bin\{Debug,Release}\kp.exe -c MALP.cfg
\endverbatim
This will simulate the first ten seconds after the breach, and write each second to a file named 
MALP_ZB_dx15.nc<time>.pgm
where <time> is the simulation time. These are pgm-files that can be opened e.g., by the GIMP, a free
image manipulation program. You can download the GIMP from http://gimp.org
This simulation took its parameters from the file MALP.cfg. Try opening MALP.cfg in notepad, and try 
changing some of the parameters and rerun the simulation. You can see all supported options and their 
supported values by executing
\verbatim
..\..\KPSimulatorTest\build\bin\{Debug,Release}\kp.exe -h
\endverbatim

\subsection Visualizer
It is also possible to run the real-time-visualizer that uses OpenGL. The visualizer supplied
has been created for academic purposes, and is only a simple prototype.

\subsection commands Visualizer commands
This visualizer has been created for research,
presentations and publications. It is undocumented, and with awkward key
bindings. Here are a few pointers on how to use it:
- pgup - Increase camera speed
- pgdown - Decrease camera speed
- f1 - create screenshots
- f2 - record path
- ctrl+s - save state
- ctrl+o - open state
- ctrl+q - quit
- esc - quit
- w - move forward
- a - move left
- s - move backward
- d - move right
- q - move up
- e - move down
- k - look down
- i - look up
- j - look left
- l - look right
- space - start/stop simulation

\section examples Running examples
Go to
\verbatim
CUDASim\data\malpasset
\endverbatim
Try running it with the following parameters:
\verbatim
..\..\KPSimulatorTest\build\bin\{Debug,Release}\kp_visualization.exe --bathymetry_no 0 --water_elevation_no 0
--nx 100 --ny 100 --width 100 --height 100 --vertical_scale 100 --time_integrator 0 --scale 0.1
--desingularization_eps 0.1
\endverbatim
The visualizer supports the same options as kp.exe, and you can also supply an image to drape over the
topography. See MALP_vis.cfg for an example:
\verbatim
..\..\KPSimulatorTest\build\bin\{Debug,Release}\kp_visualization.exe -c MALP_vis.cfg
\endverbatim
*/
