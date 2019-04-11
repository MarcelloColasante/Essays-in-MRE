%%% Initialization script for the library
str = genpath(pwd);
str_clean = [];
inds = strfind(str,';');
startind = 1;
for i = 1:length(inds)
    tmpstr = str(startind:inds(i));
    if isempty(strfind(tmpstr,'.svn'));
        str_clean = [str_clean tmpstr]; %#ok<AGROW>
    end
	startind = inds(i)+1;
end
addpath(str_clean);