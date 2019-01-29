clear;
clc;

%% reading trained weight values
w = xlsread('trainedW.xlsx');

%% fourier properties
n=2;
d=2;
%phi = zeros(11,(n+1)^d);
c = [0,0;1,0;0,1;1,1;2,0;0,2;2,2;2,1;1,2];

%% action value matrix
% we are calculating 10 different action values for age from 45 to 53, as
% there are 11 possible action, q matrix will be 11x10 for an age
q = zeros(11,10,5);

%% mapping policy for age = [45,47,49,51,53]

x_source = ones(10,10);
x_source(2,:) = (0.1:0.1:1);
for i=4:2:10
    x_source(i,:) = x_source(2,:);
end
for i=1:2:9
    x_source(i,:) = x_source(i,:)*(0.45 + (i-1)/100);
end

%% feeding input to fourier function
for j = 1:2:9    
    for i=1:10
        x = x_source(j:j+1,i);
        phi = cos(pi*c*x);
        q(:,i,((j-1)/2)+1) = w*phi;
    end
end

%% plotting the heat map
for i = 1:5
    figure
    surf(q(:,:,i))
    colorbar
    ylabel('Actions');
    xlabel('Percentage of budget remaining');
    zlabel('State-Action Values');
    formatSpec = 'Age = %d';
    str = sprintf(formatSpec,[45 + (i-1)*2]);
    title(str)
end
