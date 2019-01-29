clear;
clc;

%% Reading data
data = xlsread('InputData_Environment.xlsx');
steady_pop = data(:,1:9); % steady state population
inc = data(:,11); % incidence of cancer

%% Problem Structure
% currently taking the percentage of budget lefover and
% age as the state space of MDP
% e.g. 50% is left at age 30 or 60% left at age 70
% then the action percentage increase in diagnostic rates in that specific
% age for all pre-clinical states. But this doesn't cover everything that
% we need to cover. Basically when we decisde to screen, we'll also include
% the screening done on healthy people. This will cover the wastage of
% screening. To model this wastage we just need to collect the steady state
% out flow from healthy state due to screening

%% Starting state
% agent will always start from 100% of available budget and age=1, as the
% budget percentage number can be continuous, therefore I am using Fourier
% function approximation of order 3 to approximate the state space
budget = 7*10^6;
s = [1,0.01];

%% Starting action
% randomly choosing the initial action
a_space = (0:0.1:1);
a_c_idx = randi(size(a_space,2));

%% Weight matrix
% initializing weight matrix
w = zeros(size(a_space,2),9);

%% Episode reward and Returns matrix
trials_total = 20;
episodes_total = 50000;
episode_reward = 0;
returns = zeros(1,episodes_total);
r_trials = zeros(20,episodes_total);

%% Hyper-parameters
epsilon = 0.1;
alfa = 0.001;
gamma = 0.9778;
count = 0;

for trials = 1:trials_total
    
    %% initializing returns matrix
    returns = zeros(1,2000);
   
    %% initializing weights
    w = zeros(size(a_space,2),9);
    
    for epi_n = 1:episodes_total

        while s(1,1) > 0

            while s(1,2) < 1
                %% Observing next state from the environment           
                % Taking action 'a' in current state is equivalent to setting a
                % goal for increment in diagnosed cases of cancer in age = s(1,2).
                % Goal, in current problem formulation, is (100*a)% increment in
                % incidence of cancer for age = s(1,2). In doing so, we'll be
                % required to allocate some amount of budget. This amount invested
                % will generate additional (datum values are taken from steady pop)
                % diagnosed cases or will not. Therefore, in a way, additional
                % diagnosed cases is measure of effectiveness of investment. Hemce,
                % allocated budget and diagnosed cases are considered in reward
                % calculation.
                pop_age = steady_pop(round(100*s(1,2)),:);
                inc_age = inc(round(100*s(1,2)),:);
                [s_next,reward] = environment(s,a_space(1,a_c_idx),pop_age,inc_age,budget);

                %% fourier approximation
                % now we have information reguarding current state, action and next
                % state, we'll feed this information to fourier function which will
                % approximate the current and next state space to a feature vector.
                % Using this feature vectors (for current and next state) and
                % weight matrix we will calculate the TD-Error.
                [phi_sa_c,estimate,a_next_max_idx,target,episode_reward] = fourier(s,s_next,w,a_c_idx,a_space,gamma,episode_reward,reward);

                %% Learning
                % fourier function is returning estimate and target; essentially
                % the variables responsible for TD-Error
                delta = (target - estimate);
                w = w + alfa * delta * phi_sa_c;

                %% Updates
                s = s_next;
                budget = budget*(s_next(1,1));

                %% epsilon greedy action selection
                if rand() <= epsilon
                    % Explore
                    a_c_idx = randi(size(a_space,2));
                else
                    % Exploit
                    a_c_idx = a_next_max_idx;
                end

                %% episode reward
                episode_reward = episode_reward;

                %% count
                count = count+1;

            end

            if s(1,2) >= 1
                break
            end
        end

        %% collecting the return
        returns(1,epi_n) = episode_reward;

        %%
        count = 0;

        %% state action initialization
        s = [1,0.01];
        a_c_idx = randi(size(a_space,2));
        episode_reward = 0;
    end
    
    %% storing returns data and
    r_trials(trials,:) = returns;
end    

%% plotting average returns
avg_r = mean(r_trials,1);
figure
plot(avg_r);
formatSpec = 'Epsilon = %d, Alfa = %d, Gamma = %d';
str = sprintf(formatSpec,[epsilon,alfa,gamma]);
title(str)
xlabel('Episodes')
ylabel('Average returns')

%% extracting the optimal policy
% please refer policy map




