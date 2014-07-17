function write_pcd(pcloud, writename)

  assert(size(pcloud, 2)==3 || size(pcloud, 2)==6);
  assert(ndims(pcloud)==2);
  
  fid = fopen(writename,'w');
  fprintf(fid, '%s\n', '# .PCD v.7 - Point Cloud Data file format');
  
  if size(pcloud, 2)==3
    fprintf(fid, '%s\n', 'FIELDS x y z');
    fprintf(fid, '%s\n', 'SIZE 4 4 4');
    fprintf(fid, '%s\n', 'TYPE F F F');
    fprintf(fid, '%s\n', 'COUNT 1 1 1');
    
  elseif size(pcloud, 2)==6
    
    % converting rgb to packed float
    
    
    fprintf(fid, '%s\n', 'FIELDS x y z rgb');
    fprintf(fid, '%s\n', 'SIZE 4 4 4 4');
    fprintf(fid, '%s\n', 'TYPE F F F F');
    fprintf(fid, '%s\n', 'COUNT 1 1 1 1');
    
  end
  
  if length(pcloud) == 480*640
    fprintf(fid, '%s\n', 'WIDTH 640');
    fprintf(fid, '%s\n', 'HEIGHT 480');
  else
    fprintf(fid, '%s\n', ['WIDTH 1']);
    fprintf(fid, '%s\n', ['HEIGHT ' num2str(length(pcloud))]);
  end
  fprintf(fid, '%s\n', 'VIEWPOINT 0 0 0 1 0 0 0');
  fprintf(fid, '%s\n', ['POINTS ' num2str(length(pcloud))]);
  fprintf(fid, '%s\n', 'DATA ascii');

  %for j = 1:length(pcloud)
  %  fprintf(fid, '%f %f %f\n', pcloud(1, j), pcloud(2, j), pcloud(3, j));
  %end
  fclose(fid);
  
  dlmwrite(writename, pcloud, 'delimiter', ' ', '-append')
  
  


end
