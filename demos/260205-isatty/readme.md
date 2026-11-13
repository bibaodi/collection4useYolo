# Test different scenarios
./test_tty                    # isatty=1 (terminal)
./test_tty | cat              # isatty=0 (pipe)
./test_tty > output.txt       # isatty=0 (regular file)
./test_tty 2>&1 | cat         # isatty=0 for stdout, but stderr also piped
