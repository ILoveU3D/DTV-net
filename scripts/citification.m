clc;clear;
root = "/media/wyk/wyk/Data/result/";
% root = "./reuslt/";
origin = "origin";
compare = ["SIRT", "FISTA", "DTVCP", "FISTAnet", "Ours"];
items = ["pa_56.raw","pa_113.raw","pa_223.raw","pa_441.raw"];
citi = @(x, y) ssim(x,y);

for med = 1:length(compare)
    values = zeros([length(items), 1]);
    for itm = 1:length(items)
        path_x = sprintf("%s%s/%s", root, compare(med), items(itm));
        path_y = sprintf("%s%s/%s", root, origin, items(itm));
        x = open_file(path_x);
        y = open_file(path_y);
        x = x ./ max(y);
        y = y ./ max(y);
        values(itm) = citi(x,y);
    end
    fprintf("%s: 【%f】\n", compare(med), mean(values));
end

function file = open_file(path)
fileID = fopen(path, "r"); % Open the current proj file
file = fread(fileID,'float');
fclose(fileID);
end