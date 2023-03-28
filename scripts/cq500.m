clc;clear;
root = "/media/wyk/wyk/Data/cq500/cq500";
target = "/media/wyk/wyk/Data/raws/trainDataCq500";
expris = dir(root);
expris = expris(3:end);
step = 0;
for i = 1:length(expris)
    p1 = sprintf("%s/%s", root, expris(i).name);
    unknown = dir(p1);
    unknown = unknown(3);
    p2 = sprintf("%s/%s/%s", root, expris(i).name, unknown.name);
    studys = dir(p2);
    studys = studys(3:end);
    for j = 1:length(studys)
        p3 = sprintf("%s/%s/%s/%s", root, expris(i).name, unknown.name, studys(j).name);
        dcms = dir(p3);
        dcms = dcms(3:end);
        for k = 1:floor(length(dcms)/64)-1
            raws = zeros(256,256,64);
            for s = 1:64
                filene = sprintf("%s/%s",p3,dcms(k*64+s).name);
                img = dicomread(filene);
                img = imresize(double(img), [256,256]);
                img = max(img, 0);
                raws(:,:,s) = img';
            end
            tne = sprintf("%s/%d.raw", target, step);
            save_file(tne, raws);
            step = step + 1;
            fprintf("%s:%d generated!\n", p3, k);
        end
    end
end

function save_file(path, img)
fileID = fopen(path, 'wb+'); % Save the new proj data
fwrite(fileID, img, 'float');
fclose(fileID);
end