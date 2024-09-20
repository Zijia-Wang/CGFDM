function v = cal_rup_v(t, steph)

[m, n] = size(t);

d = 1;
i = (d+1):(m-d);
j = (d+1):(n-d);

v  = zeros(m, n);
sx = zeros(m, n);
sy = zeros(m, n);

sx(i,j)=(t(i+d,j+d)-t(i-d,j+d)+t(i+d,j-d)-t(i-d,j-d))/4/steph/d;
sy(i,j)=(t(i+d,j+d)-t(i+d,j-d)+t(i-d,j+d)-t(i-d,j-d))/4/steph/d;
%sx(i,j)=(t(i+d,j)-t(i-d,j))/(2*d*steph);
%sy(i,j)=(t(i,j+d)-t(i,j-d))/(2*d*steph);

%for ii = i-0:i+0
%for jj = j-0:j+0
%sx(i,j)=sx(i,j)+(t(i+d,j)-t(i-d,j))/(2*d*steph)/1.0;
%sy(i,j)=sy(i,j)+(t(i,j+d)-t(i,j-d))/(2*d*steph)/1.0;
%end
%end

s = sqrt(sx.^2+sy.^2);

v(i,j) = 1./(s(i,j)+1e-6);

end

%% function v = cal_rup_v_avg(u, steph)
%% 
%% [m, n] = size(u);
%% 
%% d=5;
%% i = (d+1):(m-d);
%% j = (d+1):(n-d);
%% 
%% v = zeros(m, n);
%% vx = zeros(d, d, m, n);
%% vy = zeros(d, d, m, n);
%% 
%% %vx(i,j)=(u(i+1,j+1)-u(i+1,j-1)+u(i-1,j+1)-u(i-1,j-1))/4/steph;
%% %vy(i,j)=(u(i+1,j+1)+u(i+1,j-1)-u(i-1,j+1)-u(i-1,j-1))/4/steph;
%% for dx=1:d
%% for dy=1:d
%% vx(dx,dy,i,j)=(u(i+dx,j)-u(i-dx,j))/(2*d*steph);
%% vy(dx,dy,i,j)=(u(i,j+dy)-u(i,j-dy))/(2*d*steph);
%% end
%% end
%% 
%% vxa = squeeze(mean(mean(vx)));
%% vya = squeeze(mean(mean(vy)));
%% 
%% v(i,j) = 1./sqrt(vxa(i,j).^2+vya(i,j).^2+1e-6);
%% 
%% end
