#!/bin/tcsh

# Set up conda and gurobi environment

conda activate /usr/local/usrapps/infews/group_env
module load gurobi
source /usr/local/apps/gurobi/gurobi810/linux64/bin/gurobi.sh

# Submit multiple jobs at once
        
set folNameBase = Exp
set modelName = 125simple25_


foreach NN ( r45coolss3_  r45coolss5_  r45hotss3_  r45hotss5_  r85coolss3_  r85coolss5_  r85hotss3_  r85hotss5_ )

	foreach UC ( base_  stdd_  high_  ultra_ )

		foreach TC ( `seq 2020 1 2099` )


			set dirName = ${folNameBase}${NN}${UC}${modelName}${TC}
            cd $dirName

			if ($modelName == 125simple25_) then

				# Submit LSF job for the directory $dirName
   				bsub -n 8 -R "span[hosts=1]" -R "rusage[mem=20GB]" -W 5000 -x -o out.%J -e err.%J "python wrapper_simple.py"
				# Go back to upper level directory
    				cd ..

			endif

		end
	end
end

conda deactivate
