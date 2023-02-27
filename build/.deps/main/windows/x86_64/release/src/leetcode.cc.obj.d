{
    files = {
        [[src\leetcode.cc]]
    },
    depfiles_gcc = "leetcode.o: src\\leetcode.cc src\\leetcode.h\
",
    values = {
        "gcc",
        {
            "-m64",
            "-fexceptions"
        }
    }
}