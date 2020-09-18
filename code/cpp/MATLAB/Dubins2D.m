
% Script for testing the 2D Dubins dynamic stuff

%close all
clear all
clc


% state = [ XE, YE, theta, V ]
X0 = [ 0, 0, 0, 1 ];
turn_rate = 0.2;
speed_change = 0.1;

dt = 0.1;
tf = 50.0;

X = X0;

%% Integration
for ii = 1:tf/dt
    
    % Extract Variables
    XE = X(end,1);
    YE = X(end,2);
    theta = X(end,3);
    V = X(end,4);
    
    % Calculate Xdot vector
    XE_dot = V*cos(theta);
    YE_dot = V*sin(theta);
    theta_dot = turn_rate;
    V_dot = speed_change;
    
    XE    = XE    +    XE_dot*dt;
    YE    = YE    +    YE_dot*dt;
    theta = theta + theta_dot*dt;
    V     = V     +     V_dot*dt;
    
    X(end+1,:) = [ XE, YE, theta, V ];
end

XE = X(:,1);
YE = X(:,2);
theta = X(:,3);
V = X(:,4);

t = 0:dt:dt*(numel(V)-1);


%% Plots
figure(1); clf; hold all;
subplot(1,2,1); hold all; grid on; grid minor; ...
    plot(XE,YE); ...
    axis equal; ...
    xlabel('X [ m ]'); ylabel('Y [ m ]');
subplot(2,2,2); hold all; grid on; grid minor; ...
    plot(t,theta*57.7); ...
    xlabel('Time [ s ]'); ylabel('theta [ deg ]');
subplot(2,2,4); hold all; grid on; grid minor; ...
    plot(t,V); ...
    xlabel('Time [ s ]'); ylabel('V [ m/s ]');

