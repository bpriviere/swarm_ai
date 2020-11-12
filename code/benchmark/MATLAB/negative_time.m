% Script for testing integration in negative time
%close all
clear all
clc

t = linspace(-10,10);
x0 = 0;
v0 = 0;
a = 1;

x1 = x0 + v0.*t + 0.5.*a.*t.*t;
x2 = x0 + v0.*t + 0.5.*a.*t.*t.*sign(t);


figure(1); clf; hold all; grid on; grid minor; ...
    plot(t,x1); ...
    plot(t,x2); ...
    xlabel('Time [ s ]'); ylabel('Position [ m ]');
