/* Small native launcher for the project-local macOS app bundle. */
#include <mach-o/dyld.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char executable[PATH_MAX], resolved[PATH_MAX];
    uint32_t size = sizeof(executable);
    if (_NSGetExecutablePath(executable, &size) || !realpath(executable, resolved)) return 1;
    for (int i = 0; i < 4; i++) {
        char *slash = strrchr(resolved, '/');
        if (!slash) return 1;
        *slash = '\0';
    }
    if (chdir(resolved)) return 1;
    execl(".venv/bin/python", "STEM Studio", "app.py", (char *)NULL);
    perror("STEM Studio: Python launch failed");
    return 1;
}
