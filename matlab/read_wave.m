function [vx, vy, vz] = read_wave(parfile, dirname, stations)
% extract seismograms for each receiver station
% @zqhe, 2022-08
% INPUT:
%       parfile: parameter file 'params.json'
%       dirname: file path to 'wave_xy_mpi000001.nc'
%       stations: station coordinates nsta x 3 array,
%                     x, y, z coordinate index
% 

par = get_params(parfile);
NX = par.NX;
NY = par.NY;
TMAX = par.TMAX;
Tskip = par.EXPORT_TIME_SKIP;
DT = par.DT;
PX = par.PX;
PY = par.PY;
PZ = par.PZ;

ni = NX/PX;
nj = NY/PY;

pk = PZ - 1;

% nt = floor(TMAX/DT/Tskip);
nt = floor(TMAX/DT/Tskip)-1;
nsta = size(stations,1);
vx = zeros(nt,nsta);
vy = zeros(nt,nsta);
vz = zeros(nt,nsta);

[X,Y,~] = gather_coord_xy(parfile);

X1 = X(1,:);
Y1 = Y(:,1);

for ista = 1:nsta
    % locate station index and file
    [~,gi] = min(abs(X1 - stations(ista,1)));
    [~,gj] = min(abs(Y1 - stations(ista,2)));
    
    pi = floor((gi(ista) - 1) / ni);
    pj = floor((gj(ista) - 1) / nj);
    i = gi(ista) - pi * ni;
    j = gj(ista) - pj * nj;
    
    fname = [dirname,'/wave_xy_mpi',...
            num2str(pi,'%02d'),...
            num2str(pj,'%02d'),...
            num2str(pk,'%02d'),'.nc'];
    %disp(ista)
    %disp(fname)
    
    vx(:,ista) = squeeze(ncread(fname, 'Vx', [j, i, 1], [1, 1, Inf]));
    vy(:,ista) = squeeze(ncread(fname, 'Vy', [j, i, 1], [1, 1, Inf]));
    vz(:,ista) = squeeze(ncread(fname, 'Vz', [j, i, 1], [1, 1, Inf]));
end
end