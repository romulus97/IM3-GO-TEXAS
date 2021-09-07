#!/bin/tcsh

# Set up conda and gurobi environment

conda activate /usr/local/usrapps/infews/CAPOW_env
module load gurobi
source /usr/local/apps/gurobi/gurobi810/linux64/bin/gurobi.sh

# Submit multiple jobs at once

set folNameBase = Exp

foreach NN ( 75 100 125 150 175 200 225 250 275 300 )

	foreach UC ( _simple_ _coal_ )

		foreach TC ( 25 50 75 100 )

			set dirName = ${folNameBase}${NN}${UC}${TC}
   			cd $dirName
			
			if ($UC == _simple_) then

				# Submit LSF job for the directory $dirName
   				bsub -n 8 -R "span[hosts=1]" -R "rusage[mem=20GB]" -W 5000 -x -o out.%J -e err.%J "python wrapper_simple.py"
				# Go back to upper level directory
    				cd ..

			else if ($UC == _coal_) then

				# Submit LSF job for the directory $dirName
   				bsub -n 8 -R "span[hosts=1]" -R "rusage[mem=20GB]" -W 5000 -x -o out.%J -e err.%J "python wrapper_coal.py"
				# Go back to upper level directory
    				cd ..

			endif

		end
	end
end

conda deactivate