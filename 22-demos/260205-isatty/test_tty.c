#include <unistd.h>
#include <stdio.h>
int main() {
    printf("isatty(stdout) = %d\n", isatty(1));
    return 0;
}
