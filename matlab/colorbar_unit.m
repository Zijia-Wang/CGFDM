function colorbar_unit(str)
%h = colorbar;
%set(get(h,'title'),'string',str);

hco = colorbar ;
%set(hco,'YTick',0:0.1:1);
t = get(hco,'YTickLabel');
t = strcat(t,str);
set(hco,'YTickLabel',t);
end
