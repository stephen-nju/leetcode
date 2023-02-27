{
    values = {
        "gcc",
        {
            "-m64"
        }
    },
    files = {
        [[src\leetcode.cc]]
    },
    depfiles_gcc = "leetcode.o: src\\leetcode.cc src\\leetcode.h\
"
}