function [phi_sa_c,estimate,a_next_max_idx,target_q,episode_reward] = fourier(s_current,s_next,w,a_c_idx,a_space,gamma,episode_reward,reward)
    
    n=2;
    d=2;
    phi_sa_c = zeros(size(a_space,2),(n+1)^d);
    c = [0,0;1,0;0,1;1,1;2,0;0,2;2,2;2,1;1,2];
    %%
    x = zeros(2,1);
    x(2,1) = s_current(1,1);
    x(1,1) = s_current(1,2);
    
    phi_sa_c(a_c_idx,:) = cos(pi*c*x);
    estimate = w(a_c_idx,:)*phi_sa_c(a_c_idx,:)';    
    
    %% next action selection
    x(2,1) = s_next(1,1);
    x(1,1) = s_next(1,2);
    phi_sa_n = cos(pi*c*x);
    dot_next = w*(phi_sa_n);
    [dot_max,a_next_max_idx] = max(dot_next);
    %a_next_max = a_space(a_next_max_idx,1);
    
    %% for multiple optimal actions
    check_1 = (dot_max==dot_next);
    check = sum(check_1);
    if check >= 2
         indices = find(check_1);
         a_next_max_idx = indices(randi(size(indices,1)),1);
         a_next_max = a_space(1,a_next_max_idx);
         dot_max = dot_next(a_next_max_idx,1);
    end
    
    %%
    % checking reward
    r_t = reward;
    
    % calculation episode reward
    episode_reward = episode_reward + r_t;

    % calculating target value
    target_q = gamma*dot_max + r_t;
    
end