clear;
clc;

x1 = (-10:0.1:10);
x2 = (-10:0.1:10);
y = zeros(201,201);
for i=1:201
    for j=1:201
        y(i,j) =  (x1(i).^3 - x2(j)).^2 + 2*(x2(j) - x1(i)).^4;
    end
end

contour(x1,x2,y,'ShowText','on');