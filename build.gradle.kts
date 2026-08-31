plugins {
    kotlin("jvm") version "2.3.0"
    application
}

group = "io.github.lxp-technologies"
version = "0.1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

val pytorchNative = providers.gradleProperty("pytorchNative").orElse("auto")

dependencies {
    implementation(platform("org.springframework.boot:spring-boot-dependencies:3.5.16"))
    implementation(platform("ai.djl:bom:0.36.0"))
    implementation("ai.djl:api")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.20.0")
    implementation("com.fasterxml.jackson.dataformat:jackson-dataformat-yaml:2.20.0")
    implementation("info.picocli:picocli:4.7.7")
    implementation("org.springframework.boot:spring-boot-starter-web")

    runtimeOnly("ai.djl.pytorch:pytorch-engine")
    runtimeOnly("ai.djl.pytorch:pytorch-jni:2.7.1-0.36.0")

    testImplementation(platform("org.junit:junit-bom:5.13.4"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.assertj:assertj-core:3.27.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

kotlin {
    jvmToolchain(25)
}

application {
    mainClass = "io.github.lxptechnologies.lxpmini.MainKt"
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("org.slf4j.simpleLogger.defaultLogLevel", "error")
    configurePyTorchNative(pytorchNative.get())
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("org.slf4j.simpleLogger.defaultLogLevel", "error")
    configurePyTorchNative(pytorchNative.get())
}

fun JavaForkOptions.configurePyTorchNative(choice: String) {
    when (choice) {
        "auto" -> Unit
        "cpu" -> {
            systemProperty("PYTORCH_VERSION", "2.7.1")
            systemProperty("PYTORCH_FLAVOR", "cpu")
        }
        "cuda" -> {
            systemProperty("PYTORCH_VERSION", "2.7.1")
            systemProperty("PYTORCH_FLAVOR", "cu128")
            if (System.getProperty("os.name").startsWith("Windows", ignoreCase = true)) {
                val cudaDirectory = file(
                    "${System.getProperty("user.home")}/.djl.ai/pytorch/2.7.1-cu128-win-x86_64",
                ).absolutePath
                environment("PATH", "$cudaDirectory;${System.getenv("PATH")}")
            }
        }
        else -> throw GradleException("-PpytorchNative must be auto, cpu, or cuda")
    }
}
