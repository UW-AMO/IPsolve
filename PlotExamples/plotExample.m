%clear all;

example = 'studentPL';


switch(example)
    case{'vapnik'}
        plq = 'vapnik';
        
        params.eps = 0.5;
        params.lambda = 1;
        params.silent = 1;
        
        plotPenalty(plq, params);
        
    case{'huber'}
        plq = 'huber';
        
        params.kappa = 0.2;
        plotPenalty(plq, params);
        
    case{'hybrid'}
        plq = 'hybrid';
        params.uConstraints = 1;
        params.scale =5;
        params.silent = 1;
        
        plotPenalty(plq, params);
        
        
    case{'student'}
        plq = 'student';
        params.constraints=1;
        params.uConstraints = 1;
        params.scale =5;
        params.silent = 1;
        
        plotPenalty(plq, params);

        
    case{'studentPL'}
        plq = 'studentPL';
        params.constraints=1;
        params.uConstraints = 1;
        params.kappa = 0.5;
        params.scale =1;
        params.silent = 1;
        
        plotPenalty(plq, params);

        
        
    otherwise
        error('unknown example');
end