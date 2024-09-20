function snap = gather_snap_wave_xy(parfile, dirnm, var, it)
par = get_params(parfile);

NX = par.NX;
NY = par.NY;
NZ = par.NZ;
DH = par.DH;
PX = par.PX;
PY = par.PY;
PZ = par.PZ;

ni = NX/PX;
nj = NY/PY;
nk = NZ/PZ;

snap = zeros(NY,NX);

%dirnm = par.OUT;

k = PZ-1;
for i = 0:PX-1
for j = 0:PY-1

fnm = [dirnm, '/wave_xy_mpi',...
num2str(i,'%02d'),...
num2str(j,'%02d'),...
num2str(k,'%02d'),'.nc'];

j1 = j * nj + 1; j2 = j1 + nj-1;
i1 = i * ni + 1; i2 = i1 + ni-1;

if(nargin < 4)
v = ncread(fnm, var, [1 1], [nj ni]);
else
v = ncread(fnm, var, [1 1 it], [nj ni 1]);
v = squeeze(v);
end

snap(j1:j2,i1:i2) = v;

end
end

end
