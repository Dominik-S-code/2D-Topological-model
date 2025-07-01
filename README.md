# 2D-Topological-model
A model for imaging scrambling in 2D topological insulators.

How to use the program:

Disclaimer: All the commands are only tested and confirmed to work on Linux so far. Windows users may need to modify them accordingly.

The program requires numpy, matplotlib and torch python libraries. It is run through the command line by writing the name of the main file followed by the correct positional arguments and options. The list of the arguments and options  can be displayed through the --help option like so:
	
~	python3 2D_Kitaev_model.py --help
	
Which returns the following output:	
|	usage: 2D_Kitaev_model.py [-h] [--generator {cpu,torch}] [--device DEVICE]
|                         [--torch_batch]
|                         {image,video} ...
|
|	Generate scrambling results.
|
|	positional arguments:
|	  {image,video}         Mode of operation: 'image' or 'video'.
|	    image               Generate a single image or a series of images.
|	    video               Generate a video.
|
|	options:
|	  -h, --help            show this help message and exit
|	  --generator {cpu,torch}, -g {cpu,torch}
|		                Acceleration mode: 'cpu' or 'torch'.
|	  --device DEVICE, -d DEVICE
|		                With torch generator. Device to use (e.g. 'cuda' or
|		                'cpu').

	
Writing the option --help after the arguments will display the options controlling the parameters of the simulated system, e.g.:

~	python3 2D_Kitaev_model.py -g torch -d cuda image --help

Which gives:
|	usage: 2D_Kitaev_model.py image [-h] [--system_size SYSTEM_SIZE] [--time TIME] [--filename FILENAME] [--range RANGE RANGE RANGE]
|
|	options:
|	  -h, --help            show this help message and exit
|	  --system_size SYSTEM_SIZE, -n SYSTEM_SIZE
|		                System size.
|	  --time TIME, -t TIME  Time at which to generate the image.
|	  --filename FILENAME   Filename to save the image to.
|	  --range RANGE RANGE RANGE, -r RANGE RANGE RANGE
|		                Range of time: start stop jump.

The --system_size flag takes a single number, which will be the side length of the resulting square lattice.
The --time flag chooses the time at which we measure the system.
Using the --range option instead of --time will generate a series of images instead of a single image.

All of the options have default values, but it is recommended to specify the system_size and time or range. The filename is purely optional.

An example execution of the program looks like this:

~	python3 2D_Kitaev_model.py -g torch -d cuda -b image -r 1 90 2 -n 20

This command will create 45 images starting at time 1 and ending at 89 with a jump of 2 for a 20x20 system using the torch generator. The calculations will be performed using the graphics card and accelerated using the batch generator.


Notes and possible issues:
- The program is untested on hardware other than NVidia - the cuda option is not guaranteed to work. If it causes problems, change the generator to cpu.
- Note that there's an option called 'cpu' after both --generator and --device flags. They mean different things! It is highly recommended to use the torch generator in all situations and highly preferable to use the cuda device if it works. Otherwise use the torch generator with the cpu device.
- The --torch_batch (-b) option is available if the torch generator with the cuda option is chosen. It accelerates the calculations at the cost of using significant amounts of VRAM. It also limits the size of the system that can be generated - if it is too big, the program will return an "insufficient memory" error. If this happens, reduce the size of the system or remove the --torch_batch (-b) option. During testing, a graphics card with 12 GB of VRAM can perform calculations of the system up to the size of around 20x20.
- When using the video option it may be useful to specify the --fps option (default is 15).
- The input parameters of the system not manipulated by the command line options can be changed by modifying the Input.py file, especially the j00 variable containing the coordinates of the initial perturbation.
- The outputs will be saved in the 'out' folder. The videos and solitary images will be saved to the folder directly while series of images will have separate folders created within the 'out' directory.
- The Hamiltonian can be modified to make an inhomogenous system, add disorder or create multiple neighboring systems. In this case, the line 
# HK2D = self._modify_Hamiltonian(HK2D, input) 
in the Initializer_cpu.py file must be uncommented and the _modify_Hamiltonian function must be modified accordingly to the desired changes.
