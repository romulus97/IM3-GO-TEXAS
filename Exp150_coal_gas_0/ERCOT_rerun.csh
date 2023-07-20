#!/bin/tcsh

# Set up conda and gurobi environment

conda activate /usr/local/usrapps/infews/group_env
module load gurobi
source /usr/local/apps/gurobi/gurobi810/linux64/bin/gurobi.sh

# Submit multiple jobs at once

set folNameBase = Exp

foreach NN ( 50 75 100 125 150 175 200 225 250 275 300 )
	foreach UC ( _coal_)# _coal_ )

		foreach TC ( 0 25 50 75 100 )
			set dirName = ${folNameBase}${NN}${UC}${TC}
   			cd $dirName
			
			if (-f duals.csv) then

				# Do nothing
   				:
				# Go back to upper level directory
    					cd ..

			else

				# Submit LSF job for the directory $dirName
   				bsub -n 8 -R "span[hosts=1]" -R "rusage[mem=20GB]" -W 5000 -o out.%J -e err.%J "python wrapper_coal.py"
				# Go back to upper level directory
    					cd ..

			endif

		end
	end
end

conda deactivate
