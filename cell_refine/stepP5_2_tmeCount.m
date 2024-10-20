clear
clc
close all
src_mask = '/Volumes/yuan_lab/TIER2/artemis_lei/validation_new2/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/mask_ss1512_post_tumor15_900';

files = dir(fullfile(src_mask, '*.png'));

tableTmp = table("",0,0,0,0,0,'VariableNames',{'ID',...
    'tumor_pix', 'necrosis_pix', 'stroma_pix', 'fat_pix', 'parenchyma_pix'});
k = length(files);
tme_pix = zeros(k, 5);
for i = 1:k
    file_name = files(i).name;
    wsi_ID = extractBefore(file_name, '_Ss1.png');
    img = double(imread(fullfile(src_mask, file_name)));
    temp = [];
    [m, n, ~] = size(img);
    mask_digit = zeros(m, n);
    mask_digit((img(:,:,1)==255 & img(:,:,2)==0 & img(:,:,3)==255)) = 1; %necrosis
    mask_digit((img(:,:,1)==128 & img(:,:,2)==0 & img(:,:,3)==0)) = 2; %tumor
    mask_digit((img(:,:,1)==255 & img(:,:,2)==255 & img(:,:,3)==0)) = 3; %stroma
    mask_digit((img(:,:,1)==128 & img(:,:,2)==128 & img(:,:,3)==0)) = 4; %fat
    mask_digit((img(:,:,1)==0 & img(:,:,2)==255 & img(:,:,3)==255)) = 5; %parenchyma

    for j = 1:5
        temp(j) = length(find(mask_digit(:)==j));
    end
    if max(temp)>0
        tme_pix(i, 1:5) = temp;  %pix

    end

    tableTmp.ID(i) = wsi_ID;
    tableTmp.tumor_pix(i) = tme_pix(i, 2);
    tableTmp.necrosis_pix(i) = tme_pix(i, 1);
    tableTmp.stroma_pix(i) = tme_pix(i, 3);
    tableTmp.fat_pix(i) = tme_pix(i, 4);
    tableTmp.parenchyma_pix(i) = tme_pix(i, 5);

end
writetable(tableTmp, '/Volumes/yuan_lab/TIER2/artemis_lei/validation_new2/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/discovery_post_tme_pix.xlsx')