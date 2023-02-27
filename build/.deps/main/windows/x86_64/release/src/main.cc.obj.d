{
    files = {
        [[src\main.cc]]
    },
    depfiles_gcc = "main.o: src\\main.cc\
",
    values = {
        "gcc",
        {
            "-m64",
            "-fexceptions"
        }
    }
}