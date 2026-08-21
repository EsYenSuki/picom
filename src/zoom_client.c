// SPDX-License-Identifier: MIT
// picom-zoom - Client for picom's zoom socket

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "Usage: %s <zoom-value>\n", argv[0]);
		fprintf(stderr, "  zoom-value: absolute (e.g., 1.5) or relative (+0.2, -0.2)\n");
		return 1;
	}

	const char *display = getenv("DISPLAY");
	if (!display) {
		display = ":0";
	}

	char safe[64];
	int j = 0;
	for (int i = 0; display[i] && j < 60; i++) {
		safe[j++] = (display[i] == '/') ? '_' : display[i];
	}
	safe[j] = '\0';

	char socket_path[108];
	snprintf(socket_path, sizeof(socket_path), "/tmp/picom-%s.sock", safe);

	int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
	if (fd < 0) {
		perror("socket");
		return 1;
	}

	struct sockaddr_un addr = {.sun_family = AF_UNIX};
	size_t path_len = strlen(socket_path);
	if (path_len >= sizeof(addr.sun_path)) {
		path_len = sizeof(addr.sun_path) - 1;
	}
	memcpy(addr.sun_path, socket_path, path_len);
	addr.sun_path[path_len] = '\0';

	char msg[64];
	snprintf(msg, sizeof(msg), "zoom %s", argv[1]);

	ssize_t sent = sendto(fd, msg, strlen(msg), 0, (struct sockaddr *)&addr, sizeof(addr));
	close(fd);

	if (sent < 0) {
		perror("sendto");
		fprintf(stderr, "Failed to send zoom command. Is picom running?\n");
		return 1;
	}

	return 0;
}
