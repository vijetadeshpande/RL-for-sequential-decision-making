function [s_next,reward] = environment(s,a,pop_age,inc_age,budget)

    % calculating distribution of healthy and diseased population
    pop_dist = zeros(1,9);
    for i=1:9
        pop_dist(1,i) = pop_age(1,i)/sum(pop_age);
    end

    % additional population undergoing screening
    pop_screen = sum(pop_age(1,1:5))*(inc_age * (1 + a));

    % additional cases envountered due to intervention
    new_clinical = sum(pop_dist(1,5:9))*pop_screen;
        
    %% calculating reward based on new cases that have been encountered
    if new_clinical == 0
        r1 = 0;
    else
        r1 = (new_clinical*10)^1.2;
    end
    
    % second element of reward based on expenditure
     r2 = -pop_screen*14000;
    
    reward = r1;
    
    %% expenditure for screening
    s_next(1,1) = ((s(1,1) + r2/budget));
    s_next(1,2) = s(1,2) + 0.01;

end