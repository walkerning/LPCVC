for i in `seq 1 3`; do
    python scripts/manipulate_model.py -q expand-dcblock ./scripts/try_expand$i.cfg
    /home/foxfi/projects/caffe_dev/build/tools/deephi_fix fix -calib_iter 0 -gpu 1 -model ./elf_results/prune_6/m_t6c_640_i192_expand$i.prototxt -weights ./elf_results/prune_6/m_t6c_640_i192_expand$i.caffemodel  -output_dir ./elf_results/prune_6/fix_results_new/m_t6c_640_i192_ex$i
    sed -i '/use_standard_std/d' ./elf_results/prune_6/fix_results_new/m_t6c_640_i192_ex$i/deploy.prototxt
    dnnc --mode normal --cpu_arch arm64 --save_kernel --prototxt ./elf_results/prune_6/fix_results_new/m_t6c_640_i192_ex$i/deploy.prototxt --caffemodel ./elf_results/prune_6/fix_results_new/m_t6c_640_i192_ex$i/deploy.caffemodel  --output_dir elf_results/prune_6/dnnc_normal_new/m_t6c_640_i192_ex$i --dcf /home/foxfi/projects/lpcvc/PytorchToCaffe/converted_results/mnasnet_100/Ultra96.dcf --net_name m_t6c_640_i192_ex$i --dump=all
    dnnc --mode debug --cpu_arch arm64 --save_kernel --prototxt ./elf_results/prune_6/fix_results_new/m_t6c_640_i192_ex$i/deploy.prototxt --caffemodel ./elf_results/prune_6/fix_results_new/m_t6c_640_i192_ex$i/deploy.caffemodel  --output_dir elf_results/prune_6/dnnc_debug_new/m_t6c_640_i192_ex$i --dcf /home/foxfi/projects/lpcvc/PytorchToCaffe/converted_results/mnasnet_100/Ultra96.dcf --net_name m_t6c_640_i192_ex$i --dump=all
done
