function [v, t] = get_fault_seismo(parfile, dirnm, var, y, z, nfault)
par = get_params(parfile);

NX = par.NX;
NY = par.NY;
NZ = par.NZ;
DH = par.DH;
DT = par.DT;
PX = par.PX;
PY = par.PY;
PZ = par.PZ;
TMAX = par.TMAX;
TSKIP = par.EXPORT_TIME_SKIP;

nj = NY/PY;
nk = NZ/PZ;

NT = floor(TMAX/(DT*TSKIP))-1;

[X,Y,Z] = gather_coord(parfile);

Y1 = Y(:,1);
Z1 = Z(1,:);

[~,gj] = min(abs(Y1-y));
[~,gk] = min(abs(Z1-z));

pi = 0;
pj = floor( (gj-1)/nj);
pk = floor( (gk-1)/nk);

j = gj - pj * nj;
k = gk - pk * nk;
% j = mod(gj, nj);
% k = mod(gk, nk);
fnm = [dirnm, '/fault_mpi',...
num2str(pi,'%02d'),...
num2str(pj,'%02d'),...
num2str(pk,'%02d'),'.nc'];

v = ncread(fnm, var, [j k nfault 1], [1 1 1 NT]);
v = squeeze(v);

t = (0:NT-1)*DT*TSKIP;

v = reshape(v, [], 1);
t = reshape(t, [], 1);

end
